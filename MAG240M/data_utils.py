from typing import Dict, List, Tuple

import numpy as np
import torch
from constants import (
    MAG240M_AUTHOR_FEATURES,
    MAG240M_INST_FEATURES,
    MAG240M_M2V_AUTHOR_FEATURES,
    MAG240M_M2V_INST_FEATURES,
    MAG240M_M2V_PAPER_FEATURES,
    MAG240M_PAPER_FEATURES,
)
from random_walker import RandomWalker


def collate_fn(
    batch,
    random_walkers: List[RandomWalker],
    shapes: Dict[str, List[Tuple[int, int]]],
    neighbor_feature_dims: List[int],
    m2v_neighbor_feature_dim: int,
    num_neighs: int,
):
    # Redefine memory maps for each iteration to not accumulate memory
    feats_mmaps = {
        "paper": np.memmap(
            MAG240M_PAPER_FEATURES, mode="r", shape=shapes["paper"][0], dtype=np.float16
        ),
        "author": np.memmap(
            MAG240M_AUTHOR_FEATURES,
            mode="r",
            shape=shapes["author"][0],
            dtype=np.float16,
        ),
        "institution": np.memmap(
            MAG240M_INST_FEATURES,
            mode="r",
            shape=shapes["institution"][0],
            dtype=np.float16,
        ),
    }

    m2v_feats_mmaps = {
        "paper": np.memmap(
            MAG240M_M2V_PAPER_FEATURES,
            mode="r",
            shape=shapes["paper"][1],
            dtype=np.float16,
        ),
        "author": np.memmap(
            MAG240M_M2V_AUTHOR_FEATURES,
            mode="r",
            shape=shapes["author"][1],
            dtype=np.float16,
        ),
        "institution": np.memmap(
            MAG240M_M2V_INST_FEATURES,
            mode="r",
            shape=shapes["institution"][1],
            dtype=np.float16,
        ),
    }

    nodes, features, m2v_features, labels = zip(*batch)

    all_bpgs_all_neighbors_features = [None] * len(random_walkers)
    all_bpgs_all_neighbors_weights = [None] * len(random_walkers)
    all_bpgs_all_m2v_neighbors_features = [None] * len(random_walkers)

    for i, random_walker in enumerate(random_walkers):
        single_bpg_neigh_features = [None] * len(nodes)
        single_bpg_neigh_weights = [None] * len(nodes)
        single_bpg_m2v_neigh_features = [None] * len(nodes)
        for j, node in enumerate(nodes):
            single_node_neighs_feats = [
                torch.zeros(neighbor_feature_dims[i])
            ] * num_neighs
            single_node_neighs_weights = [torch.zeros(1)] * num_neighs
            single_node_m2v_neighs_feats = [
                torch.zeros(m2v_neighbor_feature_dim)
            ] * num_neighs
            for k, (neigh, weight) in enumerate(
                random_walker.get_top_k_neighbors_for_node(node, num_neighs)
            ):
                if neigh == -1:
                    break

                single_node_neighs_feats[k] = torch.from_numpy(
                    np.asarray(
                        feats_mmaps[random_walker.node_types[-1]][neigh],
                        dtype=np.float32,
                    )
                )

                single_node_neighs_weights[k] = torch.as_tensor([weight])

                single_node_m2v_neighs_feats[k] = torch.from_numpy(
                    np.asarray(
                        m2v_feats_mmaps[random_walker.node_types[-1]][neigh],
                        dtype=np.float32,
                    )
                )

            single_bpg_neigh_features[j] = torch.stack(single_node_neighs_feats, dim=0)
            single_bpg_neigh_weights[j] = torch.stack(single_node_neighs_weights, dim=0)
            single_bpg_m2v_neigh_features[j] = torch.stack(
                single_node_m2v_neighs_feats, dim=0
            )

        all_bpgs_all_neighbors_features[i] = torch.stack(
            single_bpg_neigh_features, dim=0
        )
        all_bpgs_all_neighbors_weights[i] = torch.stack(single_bpg_neigh_weights, dim=0)
        all_bpgs_all_m2v_neighbors_features[i] = torch.stack(
            single_bpg_m2v_neigh_features, dim=0
        )

    return (
        torch.stack(features, dim=0),
        torch.stack(labels, dim=0),
        all_bpgs_all_neighbors_features,
        all_bpgs_all_neighbors_weights,
        torch.stack(m2v_features, dim=0),
        all_bpgs_all_m2v_neighbors_features,
    )
