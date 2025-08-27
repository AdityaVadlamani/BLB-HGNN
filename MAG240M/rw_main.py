import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Any, Dict, Tuple

import numpy as np
import ray
from ogb.lsc import MAG240MDataset

from constants import DATASET_DIRECTORY
from graph_utils import (GraphDict, combine_graph_dicts,
                         edge_index_to_graph_dict, get_graph_dict,
                         save_graph_dict)
from random_walker import RandomWalker


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )

    parser.add_argument("--nnodes", default=1, type=int, help="number of nodes")

    parser.add_argument("--ncpus", default=8, type=int, help="number of cpus per node")

    parser.add_argument(
        "--restart_prob",
        default=0.15,
        type=float,
        help="restart probability for random walks",
    )

    parser.add_argument(
        "--num_walks",
        default=1000,
        type=int,
        help="number of steps per node for random walks",
    )

    parser.add_argument(
        "--num_neighbors_to_store",
        default=100,
        type=float,
        help="number of neighbors to save counts for",
    )

    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="enable inter-random walk parallelism",
    )

    args = parser.parse_args()
    return args


@ray.remote(scheduling_strategy="SPREAD", memory=750 * 1024 * 1024 * 1024)
def run_random_walk_dist(
    graph_dict: GraphDict, node_types: Tuple[str], random_walk_config: Dict[str, Any]
):
    return run_random_walk(
        graph_dict=graph_dict,
        node_types=node_types,
        random_walk_config=random_walk_config,
    )


def run_random_walk(
    graph_dict: GraphDict, node_types: Tuple[str], random_walk_config: Dict[str, Any]
):
    node_counts = random_walk_config["node_counts"]
    dataset_name = random_walk_config["dataset_name"]

    graph_dict_ref = ray.put(graph_dict)

    for neigh_type in set(node_types):
        random_walker = RandomWalker(
            node_types=list(node_types) + [neigh_type], dataset_name=dataset_name
        )

        random_walker.start(
            graph_dict_ref=graph_dict_ref,
            starting_nodes=random_walk_config["starting_nodes"],
            num_splits=random_walk_config["num_splits"],
            num_walks_per_node=random_walk_config["num_walks_per_node"],
            restart_prob=random_walk_config["restart_prob"],
            node_type_change_value=node_counts[node_types[0]],
            interested_in_starting_nodes=node_types[0] == neigh_type,
            num_neighbors_to_store=random_walk_config["num_neighbors_to_store"],
        )

        print("\tSaving random walks...")
        s = time.time()
        random_walker.save_random_walks()
        print("\tSaved random walks in {} seconds".format(time.time() - s))

    del graph_dict_ref

    return 0


def main() -> None:
    args = parse_args()
    args.data = "MAG240M"

    print("Loading dataset and saving graphs if needed...")
    start_time = time.time()
    dataset = MAG240MDataset(root=os.path.join(DATASET_DIRECTORY, args.data))

    node_counts = {
        "paper": dataset.num_papers,
        "author": dataset.num_authors,
        "institution": dataset.num_institutions,
    }

    all_node_types = [("paper", "paper"), ("paper", "author"), ("paper", "institution")]

    graph_dicts = {}
    if ("paper", "paper") in all_node_types:
        paper_paper = get_graph_dict(
            dataset_name=args.data, node_types=["paper", "paper"]
        )
        if paper_paper is None:
            paper_paper = edge_index_to_graph_dict(
                dataset.edge_index("paper", "paper"),
                node_counts=node_counts,
                node_types=["paper", "paper"],
                dtype=np.uint32,
            )
            save_graph_dict(
                dataset_name=args.data,
                node_types=["paper", "paper"],
                graph_dict=paper_paper,
            )
        graph_dicts[("paper", "paper")] = paper_paper

    if ("paper", "author") in all_node_types or (
        "paper",
        "institution",
    ) in all_node_types:
        paper_author = get_graph_dict(
            dataset_name=args.data, node_types=["paper", "author"]
        )
        if paper_author is None:
            paper_author = edge_index_to_graph_dict(
                np.flip(dataset.edge_index("author", "paper"), axis=0),
                node_counts=node_counts,
                node_types=["paper", "author"],
                dtype=np.uint32,
            )
            save_graph_dict(
                dataset_name=args.data,
                node_types=["paper", "author"],
                graph_dict=paper_author,
            )
        graph_dicts[("paper", "author")] = paper_author

    if ("paper", "institution") in all_node_types:
        paper_institution = get_graph_dict(
            dataset_name=args.data, node_types=["paper", "institution"]
        )
        if paper_institution is None:
            author_institution = get_graph_dict(
                dataset_name=args.data, node_types=["author", "institution"]
            )
            if author_institution is None:
                author_institution = edge_index_to_graph_dict(
                    dataset.edge_index("author", "institution"),
                    node_counts=node_counts,
                    node_types=["author", "institution"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.data,
                    node_types=["author", "institution"],
                    graph_dict=author_institution,
                )

            paper_institution = combine_graph_dicts(
                paper_author,  # type: ignore
                author_institution,  # type: ignore
                node_counts,
                ["paper", "author", "institution"],
                dtype=np.uint32,
            )
            save_graph_dict(
                dataset_name=args.data,
                node_types=["paper", "institution"],
                graph_dict=paper_institution,
            )

            # Delete unused graph dicts arrays
            del author_institution
            if ("paper", "author") not in all_node_types:
                del paper_author  # pyright: ignore[reportUnboundVariable]

        graph_dicts[("paper", "institution")] = paper_institution

    # Only need random walk results for ArXiv papers
    starting_nodes = np.argwhere(~np.isnan(dataset.paper_label)).reshape(-1)
    print(
        f"Finished loading data and creating/saving graphs in {time.time() - start_time} seconds"
    )

    del dataset

    if not ray.is_initialized():
        if "redis_password" in os.environ:
            ray.init(
                address="auto",
                num_cpus=args.ncpus,
                _redis_password=os.environ["redis_password"],
                include_dashboard=True,
            )
        else:
            ray.init(include_dashboard=True, num_cpus=args.ncpus)

    try:
        print("Running random walks...")

        random_walk_config = {
            "dataset_name": args.data,
            "num_splits": args.nnodes * args.ncpus * 50,
            "num_walks_per_node": args.num_walks,
            "restart_prob": args.restart_prob,
            "starting_nodes": starting_nodes,
            "node_counts": node_counts,
            "num_neighbors_to_store": args.num_neighbors_to_store,
        }
        # Run the random walk for each bipartite graph
        if args.parallelize:
            ray.get(
                [
                    run_random_walk_dist.remote(
                        graph_dicts[node_types],
                        node_types=node_types,  # pyright: ignore[reportGeneralTypeIssues]
                        random_walk_config=random_walk_config,
                    )
                    for node_types in all_node_types
                ]
            )
        else:
            for node_types in all_node_types:
                run_random_walk(
                    graph_dicts[node_types],
                    node_types=node_types,
                    random_walk_config=random_walk_config,
                )

        print("Finished running random walks")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
