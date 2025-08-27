import os
import random
from collections import Counter
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import ray

from constants import RANDOM_WALK_OUTPUT_PATH
from graph_utils import GraphDict, get_neighbors_from_graph_dict
from random_walker_utils import RandomWalkDict, RandomWalkDictActor
from utils import execute_function_on_list, find_index


class RandomWalker:
    """
    Public RandomWalker object to pass to
    """

    def __init__(
        self,
        node_types: List[str],
        dataset_name: str,
    ) -> None:
        self.node_types = node_types  # list of node types as strings
        self.dataset_name = dataset_name

        self.random_walk_dict: RandomWalkDict = {}

        if not os.path.exists(RANDOM_WALK_OUTPUT_PATH):
            os.makedirs(RANDOM_WALK_OUTPUT_PATH)

    def start(
        self,
        graph_dict_ref: ray.ObjectRef,
        starting_nodes: np.ndarray,
        num_splits: int,
        num_walks_per_node: int,
        restart_prob: float,
        node_type_change_value: int,
        interested_in_starting_nodes: bool,
        num_neighbors_to_store: Optional[int] = None,
    ) -> None:
        """
        This function starts the random walks on the graph
        graph_dict_ref: ray.ObjectRef -- object ref for graph_dict
        starting_nodes: np.ndarray -- nodes to start the random walk on
        num_splits: int -- number of arrays to split starting nodes into
        num_walks_per_node: int -- number of random walks for each starting node
        restart_prob: float -- probability to which we restart
        node_type_change_value: int -- Starting node value of other type
        interested_in_starting_nodes: bool -- neighbor type to track is the same as the node type
        num_neighbors_to_store: Optional[int] -- number of neighbors to save to file for later use. If None, saves all.
        """

        assert (
            0 <= restart_prob <= 1
        ), "Expected restart probability to be between 0 and 1"

        self.save_path = os.path.join(
            RANDOM_WALK_OUTPUT_PATH,
            "{}-{}_p={}.npz".format(
                self.dataset_name, "-".join(self.node_types), restart_prob
            ),
        )

        N = len(starting_nodes)

        print(
            f"{'-'.join(self.node_types[:-1])} graph with {self.node_types[-1]} neighbors"
        )

        start_rw = datetime.now()
        print(
            "\tStarted random walk at {}".format(
                start_rw.strftime("%d-%m-%Y-%H:%M:%S"),
            )
        )

        results_ref = [
            execute_function_on_list.remote(
                self.random_walk_starting_at_node,
                starting_nodes[i * N // num_splits : (i + 1) * N // num_splits],
                graph_dict_ref,  # pyright: ignore[reportGeneralTypeIssues]
                num_walks_per_node,
                restart_prob,
                node_type_change_value,
                interested_in_starting_nodes,
                num_neighbors_to_store,
            )
            for i in range(num_splits)
        ]

        timeout = 30
        actor = RandomWalkDictActor.remote(N, num_neighbors_to_store)
        while len(results_ref) > 0:
            ready_ref, results_ref = ray.wait(
                results_ref, num_returns=len(results_ref), timeout=timeout
            )
            if ready_ref:
                ready_list = ray.get(ready_ref)
                actor.update.remote(ready_list)  # type: ignore

        self.random_walk_dict = ray.get(actor.get.remote())  # type: ignore

        end_rw = datetime.now()
        print(
            f"\tEnded random walk at {end_rw.strftime('%d-%m-%Y-%H:%M:%S')} \
                taking {(end_rw - start_rw).total_seconds()} seconds\n"
        )

    def random_walk_starting_at_node(
        self,
        node: int,
        graph_dict: GraphDict,
        num_walks_per_node: int,
        restart_prob: float,
        node_type_change_value: int,
        interested_in_starting_nodes: bool,
        num_neighbors_to_store: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        # Seed RNG to get deterministic results
        random.seed(int(node))

        if len(get_neighbors_from_graph_dict(graph_dict, node)) == 0:
            return []
        visited = Counter()
        for _ in range(num_walks_per_node):
            curr = node
            while True:
                curr_neighs = get_neighbors_from_graph_dict(graph_dict, curr)
                curr = curr_neighs[int(random.random() * len(curr_neighs))]

                if node != curr:
                    if interested_in_starting_nodes and curr < node_type_change_value:
                        visited[curr] += 1
                    elif (
                        not interested_in_starting_nodes
                        and curr >= node_type_change_value
                    ):
                        visited[curr - node_type_change_value] += 1

                if random.random() < restart_prob:
                    break

        return visited.most_common(num_neighbors_to_store)

    def get_top_k_neighbors_for_node(self, node: int, k=-1) -> np.ndarray:
        if k <= 0:
            k = len(self.random_walk_dict["neighbors"]) // len(
                self.random_walk_dict["node_idx"]
            )
        else:
            k = min(
                k,
                len(self.random_walk_dict["neighbors"])
                // len(self.random_walk_dict["node_idx"]),
            )

        try:
            idx = find_index(self.random_walk_dict["node_idx"], node)
        except ValueError:
            raise ValueError(f"Random walk results not found for node {node}")

        istart = self.random_walk_dict["neighbor_idx"][idx]
        iend = self.random_walk_dict["neighbor_idx"][idx + 1]

        return self.random_walk_dict["neighbors"][istart : min(istart + k, iend)]

    def save_random_walks(self) -> None:
        np.savez(
            self.save_path,
            neighbors=self.random_walk_dict["neighbors"],
            neighbor_idx=self.random_walk_dict["neighbor_idx"],
            node_idx=self.random_walk_dict["node_idx"],
        )

    def load_random_walks(self, p=0.15):
        self.save_path = os.path.join(
            RANDOM_WALK_OUTPUT_PATH,
            "{}-{}_p={}.npz".format(self.dataset_name, "-".join(self.node_types), p),
        )
        if os.path.exists(self.save_path):
            npzFile = np.load(self.save_path)
            self.random_walk_dict = {
                "neighbors": npzFile["neighbors"],
                "neighbor_idx": npzFile["neighbor_idx"],
                "node_idx": npzFile["node_idx"],
            }
        else:
            raise FileNotFoundError(
                f"No random walk file was found at {self.save_path}."
            )
