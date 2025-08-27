import itertools
import os
from typing import Dict, List, Literal, Union

import numpy as np

from constants import PREPROCESSED_GRAPHS_DIRECTORY

GraphDict = Dict[Union[Literal["adj_list"], Literal["adj_idx"]], np.ndarray]


def get_graph_dict(dataset_name: str, node_types: List[str]):
    file_path = os.path.join(
        PREPROCESSED_GRAPHS_DIRECTORY,
        f"{dataset_name}-{'-'.join(node_types)}-graph.npz",
    )
    if os.path.exists(file_path):
        npz_file = np.load(file_path)
        print(
            f"Number of edges and nodes for {'-'.join(node_types)}: {len(npz_file['adj_list'])}, {len(npz_file['adj_idx'])}"
        )
        return {"adj_list": npz_file["adj_list"], "adj_idx": npz_file["adj_idx"]}
    return None


def save_graph_dict(dataset_name: str, node_types: List[str], graph_dict: GraphDict):
    print(f"Saving {'-'.join(node_types)} graph dict...")
    file_path = os.path.join(
        PREPROCESSED_GRAPHS_DIRECTORY,
        f"{dataset_name}-{'-'.join(node_types)}-graph.npz",
    )
    np.savez(file_path, adj_list=graph_dict["adj_list"], adj_idx=graph_dict["adj_idx"])
    print(f"Saved {'-'.join(node_types)} graph dict")


def get_neighbors_from_graph_dict(graph: GraphDict, idx: int):
    istart = graph["adj_idx"][idx]
    iend = graph["adj_idx"][idx + 1]
    return graph["adj_list"][istart:iend]


def combine_graph_dicts(
    graph_1: GraphDict,
    graph_2: GraphDict,
    node_counts: Dict[str, int],
    node_types: List[str],
    dtype,
) -> GraphDict:
    combined_adj_list = [
        [] for _ in range(node_counts[node_types[0]] + node_counts[node_types[2]])
    ]
    for idx in range(node_counts[node_types[0]]):
        for neigh in get_neighbors_from_graph_dict(graph_1, idx):
            combined_adj_list[idx].extend(
                (
                    get_neighbors_from_graph_dict(
                        graph_2, neigh - node_counts[node_types[0]]
                    )  # get neighbors of the same node in other graph_dict
                    - node_counts[
                        node_types[1]
                    ]  # Subtract off the shift from second graph_dict
                    + node_counts[
                        node_types[0]
                    ]  # Add the shift for the combined graph_dict
                ).tolist()
            )

        combined_adj_list[idx] = list(set(combined_adj_list[idx]))  # Remove duplicates

    for idx in range(node_counts[node_types[2]]):
        for neigh in get_neighbors_from_graph_dict(
            graph_2, idx + node_counts[node_types[1]]
        ):
            combined_adj_list[idx + node_counts[node_types[0]]].extend(
                (
                    get_neighbors_from_graph_dict(
                        graph_1, neigh + node_counts[node_types[0]]
                    )
                ).tolist()
            )
        combined_adj_list[idx] = list(set(combined_adj_list[idx]))  # Remove duplicates

    return {
        "adj_list": np.array(
            list(itertools.chain.from_iterable(combined_adj_list)), dtype=dtype
        ),
        "adj_idx": np.cumsum(
            np.array([0] + [len(neighs) for neighs in combined_adj_list], dtype=dtype)
        ),
    }


def edge_index_to_graph_dict(
    edge_index: np.ndarray, node_counts: Dict[str, int], node_types: List[str], dtype
) -> GraphDict:
    edge_index_t = np.transpose(edge_index)

    has_single_node_type = (
        node_types[0] == node_types[1]
    )  # Used to handle paper-paper graph

    if has_single_node_type:
        adj_list = [[] for _ in range(node_counts[node_types[0]])]
    else:
        adj_list = [
            [] for _ in range(node_counts[node_types[0]] + node_counts[node_types[1]])
        ]

    for src, dest in edge_index_t:
        if has_single_node_type:
            adj_list[src].append(dest)
            adj_list[dest].append(src)
        else:
            adj_list[src].append(dest + node_counts[node_types[0]])
            adj_list[dest + node_counts[node_types[0]]].append(src)

    return {
        "adj_list": np.array(
            list(itertools.chain.from_iterable(adj_list)), dtype=dtype
        ),
        "adj_idx": np.cumsum(
            np.array([0] + [len(neighs) for neighs in adj_list], dtype=dtype)
        ),
    }
