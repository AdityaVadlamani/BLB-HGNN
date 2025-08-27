from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import ray

RandomWalkDict = Dict[
    Union[Literal["neighbors"], Literal["neighbor_idx"], Literal["node_idx"]],
    np.ndarray,
]


@ray.remote
class RandomWalkDictActor:
    def __init__(self, num_nodes, num_neighbors_to_store, dtype="i4"):
        self.num_neighbors_to_store = num_neighbors_to_store
        self.num_nodes = num_nodes

        self.neighbors = np.empty(
            num_nodes * num_neighbors_to_store, dtype=f"{dtype},{dtype}"
        )
        self.neighbors.fill((-1, 0))
        self.neighbor_idx = np.arange(
            0,
            (num_nodes + 1) * num_neighbors_to_store,
            num_neighbors_to_store,
            dtype=f"{dtype}",
        )
        self.node_idx = np.empty(num_nodes, dtype=f"{dtype}")

        self.idx_ptr = 0

    def update(self, arr_batch: List[List[Tuple[int, List[Tuple[int, int]]]]]):
        for arr in arr_batch:
            for node, nghs in arr:
                neigh_idx_ptr = self.idx_ptr * self.num_neighbors_to_store

                self.node_idx[self.idx_ptr] = node
                self.neighbors[
                    neigh_idx_ptr : neigh_idx_ptr
                    + min(len(nghs), self.num_neighbors_to_store)
                ] = nghs[: self.num_neighbors_to_store]

                self.idx_ptr += 1

    def get(self):
        return {
            "neighbors": self.neighbors,
            "neighbor_idx": self.neighbor_idx,
            "node_idx": self.node_idx,
        }
