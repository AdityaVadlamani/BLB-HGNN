import itertools
import os
from collections import defaultdict
from typing import Dict, Optional

import _pickle as cPickle
import lightning as L
import numpy as np
import torch
from constants import DATASET_DIRECTORY
from data_utils import NodeDataset, collate_fn, convert_heterodata_to_dataset
from dgl import load_graphs
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from random_walker import RandomWalker
from samplers import BootstrapDistributedSampler, GraphBootstrapDistributedSampler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.datasets import OGB_MAG, HGBDataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        /,
        *,
        name: str,
        seed: int,
        n_sample_multiplier: float,
        replica_set_frac: float,
        num_subsets: int,
        partition_across_replicas: bool = False,
        sampler_type: str = "uniform",
        world_size: int = 1,
        num_neighs: int = 50,
        num_workers: int = 16,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.name = name
        self.seed = seed
        self.world_size = world_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neighs = num_neighs

        self.sampler_type = sampler_type
        self.partition_across_replicas = partition_across_replicas
        self.num_subsets = num_subsets
        self.n_sample_multiplier = n_sample_multiplier
        self.replica_set_frac = replica_set_frac
        self.replica_idx = None

        # Attributes to be set in setup
        self.collate_fn = lambda batch: batch
        self.train_subset = None
        self.train_data = Dataset()
        self.val_data = Dataset()
        self.test_data = Dataset()

        self.already_called_setup = False

    @property
    def num_features(self) -> int:
        if self.name == "OGB_MAG":
            return 128
        elif self.name == "academic4HetGNN":
            return 384
        elif self.name in ["ohgbn-yelp2", "ohgbn-Freebase"]:
            return 128
        elif self.name == "IMDB":
            return 3489
        elif self.name == "DBLP":
            return 334

    @property
    def num_classes(self) -> int:
        if self.name == "OGB_MAG":
            return 349
        elif self.name == "academic4HetGNN":
            return 4
        elif self.name == "ohgbn-yelp2":
            return 16
        elif self.name == "IMDB":
            return 5
        elif self.name == "DBLP":
            return 4
        elif self.name == "ohgbn-Freebase":
            return 8

    @property
    def num_random_walks(self) -> int:
        if self.name == "OGB_MAG":
            return 7
        elif self.name in ["ohgbn-yelp2", "IMDB", "DBLP"]:
            return 6
        elif self.name == "academic4HetGNN":
            return 4
        elif self.name == "ohgbn-Freebase":
            return 13

    @property
    def labeled_node_type(self) -> str:
        if self.name == "OGB_MAG":
            return "paper"
        elif self.name == "academic4HetGNN":
            return "author"
        elif self.name == "ohgbn-yelp2":
            return "business"
        elif self.name == "IMDB":
            return "movie"
        elif self.name == "DBLP":
            return "author"
        elif self.name == "ohgbn-Freebase":    
            return "BOOK"

    @property
    def neighbor_feature_dims(self) -> int:
        if self.name == "academic4HetGNN":
            return [384, 128, 384, 640]
        elif self.name == "IMDB":
            return [3489, 3341, 3489, 3341, 3489, 128]
        elif self.name == "DBLP":
            return [334, 4231, 334, 50, 334, 128]
        else:
            return [self.num_features] * self.num_random_walks

    @property
    def is_multilabel(self) -> bool:
        return self.name in ["ohgbn-yelp2", "IMDB"]


    @property
    def train_label_weights(self) -> torch.Tensor:
        if self.train_subset is not None:
            labels, cnts = torch.unique(
                torch.cat([data[-1].unsqueeze(0) for data in self.train_subset]),
                return_counts=True,
            )

            train_label_weights = torch.zeros(self.num_classes, dtype=torch.float)
            train_label_weights[labels] = 1 - cnts / (cnts.max() + 1)
            return train_label_weights

        return None

    def prepare_data(self) -> None:
        if self.name == "OGB_MAG":  # downloads metapath2vec features for OGB_MAG
            OGB_MAG(
                os.path.join(DATASET_DIRECTORY, self.name), preprocess="metapath2vec"
            )

        # Pickle RandomWalker objects for quick loading for collate_fn
        if not os.path.exists(
            os.path.join(DATASET_DIRECTORY, self.name, "random_walkers.pickle")
        ):
            if self.name == "OGB_MAG":
                all_node_types = [
                    ["paper", "paper"],
                    ["paper", "author"],
                    ["paper", "field_of_study"],
                    ["paper", "institution"],
                ]
            elif self.name == "academic4HetGNN":
                all_node_types = [
                    ["author", "paper"],
                    ["author", "venue"],
                ]
            elif self.name == "ohgbn-yelp2":
                all_node_types = [
                    ["business", "location"],
                    ["business", "phrase"],
                    ["business", "stars"],
                ]
            elif self.name == "IMDB":
                all_node_types = [
                    ["movie", "actor"],
                    ["movie", "director"],
                    ["movie", "keyword"],
                ]
            elif self.name == "DBLP":
                all_node_types = [
                    ["author", "paper"],
                    ["author", "term"],
                    ["author", "venue"],
                ]
            elif self.name == "ohgbn-Freebase":
                all_node_types = [
                    ["BOOK", "ORGANIZATION"],
                    ["BOOK", "BOOK"],
                    ["BOOK", "LOCATION"],
                    ["BOOK", "SPORTS"],
                    ["BOOK", "FILM"],
                    ["BOOK", "BUSINESS"],
                    ["BOOK", "MUSIC"],
                ]    

            random_walkers = []
            for node_types in all_node_types:
                seen = set()
                for neigh_type in node_types:
                    if neigh_type not in seen:
                        random_walker = RandomWalker(
                            node_types=node_types + [neigh_type], dataset_name=self.name
                        )
                        random_walker.load_random_walks()
                        random_walkers.append(random_walker)
                        seen.add(neigh_type)

            with open(
                os.path.join(DATASET_DIRECTORY, self.name, "random_walkers.pickle"),
                "wb",
            ) as f:
                cPickle.dump(random_walkers, f)

    def setup(self, stage: str) -> None:
        # Prevent calling setup() multiple times unnecessarily
        if self.already_called_setup:
            return

        # Load RandomWalker objects
        with open(
            os.path.join(DATASET_DIRECTORY, self.name, "random_walkers.pickle"), "rb"
        ) as f:
            self.random_walkers = cPickle.load(f)

        # Load datasets
        if self.name == "OGB_MAG":
            dataset = OGB_MAG(
                os.path.join(DATASET_DIRECTORY, self.name), preprocess="metapath2vec"
            )[0]
        elif self.name in ["academic4HetGNN", "ohgbn-yelp2", "ohgbn-Freebase"]:
            dataset = load_graphs(
                os.path.join(DATASET_DIRECTORY, self.name, "graph.bin")
            )[0][0]
        elif self.name in ["IMDB", "DBLP"]:
            dataset = HGBDataset(os.path.join(DATASET_DIRECTORY, self.name), self.name)[
                0
            ]

        # Get counts for each node type depending if it is DGL graph or PyG heterodata
        if self.name in ["academic4HetGNN", "ohgbn-yelp2", "ohgbn-Freebase"]:
            node_counts: Dict[str, int] = {
                item: dataset.num_nodes(item) for item in dataset.ntypes
            }
            if self.name == "ohgbn-Freebase":
                node_counts.pop('PEOPLE', None)
        else:
            node_counts: Dict[str, int] = {
                str(item[0]): int(
                    item[1].x.shape[0] if "x" in item[1] else item[1].num_nodes
                )
                for item in dataset.node_items()
            }

        if self.name == "academic4HetGNN":
            data_dict = {}
            for nt, val in dataset.ndata["abstract"].items():
                if nt == self.labeled_node_type:
                    data_dict[nt] = torch.cat(
                        (
                            torch.arange(val.shape[0]).reshape(-1, 1),
                            val,
                            dataset.ndata["label"][nt].reshape(-1, 1),
                        ),
                        dim=1,
                    )
                else:
                    data_dict[nt] = torch.cat(
                        (torch.arange(val.shape[0]).reshape(-1, 1), val), dim=1
                    )

            self.all_data = NodeDataset(
                name=self.name,
                data_dict=data_dict,
                labeled_node_type=self.labeled_node_type,
                is_multilabel=self.is_multilabel,
                num_classes=self.num_classes,
            )

        elif self.name in ["ohgbn-yelp2", "ohgbn-Freebase"]:
            ndata = {}
            
            feats = "ohgbn_yelp2_m2v_feats.pt"
            
            if self.name == "ohgbn-Freebase": 
                feats = "ohgbn_freebase_m2v_feats.pt"
                
            features = torch.load(
                os.path.join(DATASET_DIRECTORY, self.name, feats)
            )

            i = 0
            for item in node_counts:
                ndata[item] = features[i : i + node_counts[item]]
                i += node_counts[item]

            data_dict = {}
            for nt, val in ndata.items():
                if nt == self.labeled_node_type:
                    label = dataset.ndata["label"][nt]
                    if self.name == "ohgbn-Freebase":
                        label = label.reshape(-1, 1)
                    data_dict[nt] = torch.cat(
                        (
                            torch.arange(val.shape[0]).reshape(-1, 1),
                            val,
                            label,
                        ),
                        dim=1,
                    )
                else:
                    data_dict[nt] = torch.cat(
                        (torch.arange(val.shape[0]).reshape(-1, 1), val), dim=1
                    )

            self.all_data = NodeDataset(
                name=self.name,
                data_dict=data_dict,
                labeled_node_type=self.labeled_node_type,
                is_multilabel=self.is_multilabel,
                num_classes=self.num_classes,
            )

        else:
            self.all_data = convert_heterodata_to_dataset(
                data=dataset,
                node_counts=node_counts,
                name=self.name,
                labeled_node_type=self.labeled_node_type,
                num_classes=self.num_classes,
                is_multilabel=self.is_multilabel,
            )

        self.collate_fn = lambda batch: collate_fn(
            batch=batch,
            dataset=self.all_data,
            random_walkers=self.random_walkers,
            neighbor_feature_dims=self.neighbor_feature_dims,
            num_neighs=self.num_neighs,
        )

        if self.name in ["academic4HetGNN", "ohgbn-yelp2", "ohgbn-Freebase"]:
            train_data = (
                dataset.ndata["train_mask"][self.labeled_node_type].nonzero().squeeze()
            )
            if dataset.ndata["val_mask"] != {}:
                val_data = (
                    dataset.ndata["val_mask"][self.labeled_node_type]
                    .nonzero()
                    .squeeze()
                )
            else:
                num_train = int(len(train_data) * 0.9)
                val_data = train_data[num_train:]
                train_data = train_data[:num_train]

            test_data = (
                dataset.ndata["test_mask"][self.labeled_node_type].nonzero().squeeze()
            )

            self.train_data = Subset(
                self.all_data,
                [(idx, self.labeled_node_type) for idx in train_data],
            )
            self.val_data = Subset(
                self.all_data, [(idx, self.labeled_node_type) for idx in val_data]
            )
            self.test_data = Subset(
                self.all_data,
                [(idx, self.labeled_node_type) for idx in test_data],
            )
            self.train_idx = train_data.numpy()
        elif self.name in ["IMDB", "DBLP"]:
            train_data = dataset[self.labeled_node_type].train_mask.nonzero().squeeze()
            if "val_mask" in dataset[self.labeled_node_type]:
                val_data = dataset[self.labeled_node_type].val_mask.nonzero().squeeze()
            else:
                num_train = int(len(train_data) * 0.9)
                val_data = train_data[num_train:]
                train_data = train_data[:num_train]

            test_data = dataset[self.labeled_node_type].test_mask.nonzero().squeeze()

            self.train_data = Subset(
                self.all_data,
                [(idx, self.labeled_node_type) for idx in train_data],
            )
            self.val_data = Subset(
                self.all_data, [(idx, self.labeled_node_type) for idx in val_data]
            )
            self.test_data = Subset(
                self.all_data,
                [(idx, self.labeled_node_type) for idx in test_data],
            )
            self.train_idx = train_data.numpy()
        else:
            self.train_data = Subset(
                self.all_data,
                [
                    (idx, self.labeled_node_type)
                    for idx in dataset[self.labeled_node_type]
                    .train_mask.nonzero()
                    .squeeze()
                ],
            )
            self.val_data = Subset(
                self.all_data,
                [
                    (idx, self.labeled_node_type)
                    for idx in dataset[self.labeled_node_type]
                    .val_mask.nonzero()
                    .squeeze()
                ],
            )
            self.test_data = Subset(
                self.all_data,
                [
                    (idx, self.labeled_node_type)
                    for idx in dataset[self.labeled_node_type]
                    .test_mask.nonzero()
                    .squeeze()
                ],
            )

            self.train_idx = dataset[self.labeled_node_type].train_mask.nonzero(as_tuple=True)[0].numpy()

        self.already_called_setup = True

    def set_replica(self, replica_idx: Optional[int] = None):
        self.replica_idx = replica_idx

    def set_train_subset(self, /, *, return_sampler: bool = False):
        reduced_n_samples = int(self.n_sample_multiplier * len(self.train_data))
        gen = np.random.default_rng(self.seed)

        sampler = None
        if "ppr" in self.sampler_type:
            if "ss" in self.sampler_type:
                ss_train_nids = np.load(
                    os.path.join(
                        DATASET_DIRECTORY, self.name, "spread_sampled_p=0.1_k=1_b=" + str(self.n_sample_multiplier) + ".npy"
                    )
                )
                assert len(ss_train_nids) == len(
                    np.intersect1d(ss_train_nids, self.train_idx)
                ), "Spread sampled nodes should only be from training set"

            nodes = defaultdict(list)
            unique = set()
            while len(unique) < reduced_n_samples:
                if "ss" in self.sampler_type:
                    seed_node = ss_train_nids[int(gen.random() * len(ss_train_nids))]
                else:
                    seed_node = self.train_idx[int(gen.random() * len(self.train_idx))]
                if seed_node not in unique:
                    unique.add(seed_node)
                    for rw in self.random_walkers:
                        if rw.node_types[-1] == self.labeled_node_type:
                            neighs = rw.get_top_k_neighbors_for_node(seed_node)
                            for neigh, _ in neighs:
                                if neigh == -1:
                                    break
                                elif neigh in self.train_idx and neigh not in unique:
                                    nodes[seed_node].append(neigh)
                                    unique.add(neigh)

            del unique
            train_subset = Subset(
                self.all_data,
                list(
                    itertools.chain.from_iterable(
                        [
                            [(node, self.labeled_node_type)]
                            + [(neigh, self.labeled_node_type) for neigh in neighs]
                            for node, neighs in nodes.items()
                        ]
                    )
                ),
            )
            weights = None
            if "weighted" in self.sampler_type:
                weights = torch.FloatTensor([len(v) for v in nodes.values()])

            if return_sampler:
                sampler = GraphBootstrapDistributedSampler(
                    train_subset,
                    seed=self.seed,
                    replica_idx=self.replica_idx,
                    num_subsets=self.num_subsets,
                    replica_set_frac=self.replica_set_frac,
                    weights=weights,
                    community_reps=np.cumsum(
                        [0] + [len(v) + 1 for v in nodes.values()]
                    ),
                )
        elif self.sampler_type == "ss":
            nodes = np.load(
                os.path.join(
                    DATASET_DIRECTORY, self.name, "spread_sampled_p=0.1_k=1_b=" + str(self.n_sample_multiplier) + ".npy"
                )
            )
            assert len(nodes) == len(
                np.intersect1d(nodes, self.train_idx)
            ), "Spread sampled nodes should only be from training set"
            nodes = [(node, self.labeled_node_type) for node in nodes]
            gen.shuffle(nodes)
            train_subset = Subset(self.all_data, nodes)
            if return_sampler:
                sampler = GraphBootstrapDistributedSampler(
                    train_subset,
                    seed=self.seed,
                    replica_idx=self.replica_idx,
                    num_subsets=self.num_subsets,
                    replica_set_frac=self.replica_set_frac,
                )
        else:
            # Randomly sample reduced_n_samples nodes from the training set
            indices = gen.choice(len(self.train_data), reduced_n_samples, replace=False)
            train_subset = Subset(self.train_data, indices)

            if return_sampler:
                sampler = BootstrapDistributedSampler(
                    train_subset,
                    seed=self.seed,
                    replica_idx=self.replica_idx,
                    num_subsets=self.num_subsets,
                    replica_set_frac=self.replica_set_frac,
                    partition_across_replicas=self.partition_across_replicas,
                )

        self.train_subset = train_subset
        if return_sampler:
            return sampler


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = self.set_train_subset(return_sampler=True)
        if self.replica_idx is None:
            # For averaged model finetuning we use the entire train_subset
            sampler = DistributedSampler(self.train_subset, shuffle=True, seed=self.seed)
        else:
            sampler.indices_setup()

        return DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
