import itertools
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import _pickle as cPickle
import lightning as L
import numpy as np
import torch
from constants import (
    DATASET_DIRECTORY,
    MAG240M_M2V_PAPER_FEATURES,
    MAG240M_PAPER_FEATURES,
)
from data_utils import collate_fn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from ogb.lsc import MAG240MDataset
from random_walker import RandomWalker
from samplers import BootstrapDistributedSampler, GraphBootstrapDistributedSampler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


class MAG240MArXivDataset(Dataset):
    def __init__(
        self,
        nodes: np.ndarray,
        features: np.ndarray,
        m2v_features: np.ndarray,
        labels: np.ndarray,
    ):
        assert len(nodes) == len(features) == len(labels)
        self.nodes = nodes
        self.features = torch.from_numpy(features).float()
        self.m2v_features = torch.from_numpy(m2v_features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(
        self, index
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        act_idx = np.argwhere(self.nodes == index)[0][0]
        return (
            index,
            self.features[act_idx],
            self.m2v_features[act_idx],
            self.labels[act_idx],
        )


class MAG240MDataModule(L.LightningDataModule):
    def __init__(
        self,
        /,
        *,
        name: str,
        seed: int,
        n_sample_multiplier: float,
        replica_set_frac: float,
        num_subsets: int,
        partition_across_replicas: float,
        sampler_type: str,
        world_size: int = 1,
        num_neighs: int = 50,
        num_workers: int = 16,
        batch_size: int = 128,
        construct_new_val_set: bool = False,
    ) -> None:
        super().__init__()

        self.name = name
        self.seed = seed
        self.world_size = world_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neighs = num_neighs
        self.construct_new_val_set = construct_new_val_set

        self.sampler_type = sampler_type
        self.partition_across_replicas = partition_across_replicas
        self.num_subsets = num_subsets
        self.n_sample_multiplier = n_sample_multiplier
        self.replica_set_frac = replica_set_frac
        self.replica_idx = None

        # Attributes to be set in setup
        self.collate_fn = lambda batch: batch
        self.train_idx = torch.tensor([])
        self.train_subset = None
        self.train_data = Dataset()
        self.val_data = Dataset()
        self.test_data = Dataset()

        self.label_weights = []

        self.already_called_setup = False

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_m2v_features(self) -> int:
        return 64

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_random_walks(self) -> int:
        return 5

    @property
    def neighbor_feature_dims(self) -> List[int]:
        return [self.num_features] * self.num_random_walks

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
        # Pickle RandomWalker objects for quick loading for collate_fn
        if not os.path.exists(
            os.path.join(DATASET_DIRECTORY, self.name, "random_walkers.pickle")
        ):
            all_node_types = [
                ["paper", "paper"],
                ["paper", "author"],
                ["paper", "institution"],
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
        if (
            self.already_called_setup
        ):  # Prevent calling setup() multiple times unnecessarily
            return

        dataset = MAG240MDataset(root=os.path.join(DATASET_DIRECTORY, self.name))
        shapes = {
            "paper": [
                (dataset.num_papers, self.num_features),
                (dataset.num_papers, self.num_m2v_features),
            ],
            "author": [
                (dataset.num_authors, self.num_features),
                (dataset.num_authors, self.num_m2v_features),
            ],
            "institution": [
                (dataset.num_institutions, self.num_features),
                (dataset.num_institutions, self.num_m2v_features),
            ],
        }

        with open(
            os.path.join(DATASET_DIRECTORY, self.name, "random_walkers.pickle"), "rb"
        ) as f:
            random_walkers = cPickle.load(f)

        self.collate_fn = lambda batch: collate_fn(
            batch,
            random_walkers,
            shapes,
            self.neighbor_feature_dims,
            self.num_m2v_features,
            self.num_neighs,
        )

        arxiv_nodes = np.argwhere(~np.isnan(dataset.paper_label)).reshape(-1)

        paper_mmap = np.memmap(
            MAG240M_PAPER_FEATURES, mode="r", shape=shapes["paper"][0], dtype=np.float16
        )
        m2v_paper_mmap = np.memmap(
            MAG240M_M2V_PAPER_FEATURES,
            mode="r",
            shape=shapes["paper"][1],
            dtype=np.float16,
        )

        arxiv_data = MAG240MArXivDataset(
            nodes=arxiv_nodes,
            features=paper_mmap[arxiv_nodes],
            m2v_features=m2v_paper_mmap[arxiv_nodes],
            labels=dataset.paper_label[arxiv_nodes],
        )
        del m2v_paper_mmap

        if self.construct_new_val_set:
            # Split training set (< 2019) into new train (< 2018) and new val (= 2018)
            train_idx = dataset.get_idx_split("train")

            self.train_idx = train_idx[
                np.where(dataset.paper_year[train_idx] < 2018)[0]
            ]
            val_idx = train_idx[np.where(dataset.paper_year[train_idx] == 2018)[0]]
            test_idx = dataset.get_idx_split("valid")

            self.train_data = Subset(arxiv_data, self.train_idx)
            self.val_data = Subset(arxiv_data, val_idx)
            self.test_data = Subset(arxiv_data, test_idx)
        else:
            self.train_idx = dataset.get_idx_split("train")

            self.train_data = Subset(arxiv_data, dataset.get_idx_split("train"))
            self.val_data = Subset(arxiv_data, dataset.get_idx_split("valid"))
            self.test_data = Subset(arxiv_data, dataset.get_idx_split("valid"))

        # self.set_train_subset()
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
                        DATASET_DIRECTORY, self.name, "spread_sampled_p=0.1_k=1.npy"
                    )
                )
                assert len(nodes) == len(
                    np.intersect1d(nodes, self.train_idx)
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
                    DATASET_DIRECTORY, self.name, "spread_sampled_p=0.1_k=1.npy"
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
            sampler = DistributedSampler(
                self.train_subset, shuffle=True, seed=self.seed
            )
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
