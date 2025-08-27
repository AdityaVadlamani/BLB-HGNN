from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from random_walker import RandomWalker
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class NodeDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_dict: Dict[str, torch.Tensor],
        labeled_node_type: str,
        is_multilabel: bool,
        num_classes: int,
    ):
        self.name = name
        self.data_dict = data_dict
        self.labeled_node_type = labeled_node_type
        self.is_multilabel = is_multilabel
        self.num_classes = num_classes

    def __len__(self) -> int:
        return sum([len(data) for data in self.data_dict.values()])

    def __getitem__(self, x: Tuple[int, Optional[str]]):

        index, type = x
        if type is None:
            data = self.data_dict[self.labeled_node_type]
        else:
            data = self.data_dict[type]

        if type == self.labeled_node_type:
            if self.is_multilabel:
                return (
                    int(data[index][0].item()),
                    data[index][1 : -self.num_classes],
                    data[index][-self.num_classes :],
                )
            return int(data[index][0].item()), data[index][1:-1], data[index][-1].long()
        else:
            return int(data[index][0].item()), data[index][1:], None


def convert_heterodata_to_dataset(
    data: HeteroData,
    node_counts,
    name: str,
    labeled_node_type: str,
    num_classes: int,
    is_multilabel: bool,
    num_features: int = 128,
    normalize: bool = False,
):
    node_type_datasets = {}
    for node_type, items in data.node_items():
        if "x" in items:
            features = items.x
            if normalize:
                features = F.normalize(features, dim=0)
        else:
            features = torch.randn((node_counts[node_type], num_features))

        if "y" in items:
            if is_multilabel:
                labels = items.y.type(torch.LongTensor).view(-1, num_classes)
            else:
                labels = items.y.type(torch.LongTensor).view(-1, 1)
            node_type_datasets[node_type] = torch.concat(
                (torch.arange(node_counts[node_type]).view(-1, 1), features, labels),
                dim=1,
            )
        else:
            node_type_datasets[node_type] = torch.concat(
                (torch.arange(node_counts[node_type]).view(-1, 1), features), dim=1
            )

    return NodeDataset(
        name=name,
        data_dict=node_type_datasets,
        labeled_node_type=labeled_node_type,
        is_multilabel=is_multilabel,
        num_classes=num_classes,
    )


def collate_fn(
    batch,
    dataset: NodeDataset,
    random_walkers: List[RandomWalker],
    neighbor_feature_dims: List[int],
    num_neighs: int,
):
    nodes, features, labels = zip(*batch)

    all_bpgs_all_neighbors_features = [None] * len(random_walkers)
    all_bpgs_all_neighbors_weights = [None] * len(random_walkers)
    for i, random_walker in enumerate(random_walkers):
        single_bpg_neigh_features = [None] * len(nodes)
        single_bpg_neigh_weights = [None] * len(nodes)
        for j, node in enumerate(nodes):
            single_node_neighs_feats = [
                torch.zeros(neighbor_feature_dims[i])
            ] * num_neighs
            single_node_neighs_weights = [torch.zeros(1)] * num_neighs
            for k, (neigh, weight) in enumerate(
                random_walker.get_top_k_neighbors_for_node(node, num_neighs)
            ):
                if neigh == -1:
                    break
                single_node_neighs_feats[k] = dataset[
                    (neigh, random_walker.node_types[-1])
                ][1]
                single_node_neighs_weights[k] = torch.as_tensor([weight])

            single_bpg_neigh_features[j] = torch.stack(single_node_neighs_feats, dim=0)
            single_bpg_neigh_weights[j] = torch.stack(single_node_neighs_weights, dim=0)

        all_bpgs_all_neighbors_features[i] = torch.stack(
            single_bpg_neigh_features, dim=0
        )
        all_bpgs_all_neighbors_weights[i] = torch.stack(single_bpg_neigh_weights, dim=0)

    return (
        torch.stack(features, dim=0),
        torch.stack(labels, dim=0).long(),
        all_bpgs_all_neighbors_features,
        all_bpgs_all_neighbors_weights,
    )
