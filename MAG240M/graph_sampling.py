import math
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import reduce

import graph_tool.all as gt
import numpy as np
import ray
from constants import DATASET_DIRECTORY
from ogb.lsc import MAG240MDataset
from utils import execute_function_on_list


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )

    parser.add_argument("--ncpus", default=8, type=int, help="number of cpus per node")

    parser.add_argument(
        "--rng_seed",
        default=0,
        type=int,
        help="RNG seed",
    )

    parser.add_argument(
        "--method",
        default="ppr",
        choices=["ppr", "spread_sampling"],
        help="sampling method to run",
    )

    parser.add_argument(
        "--target_frac",
        default=0.5,
        type=float,
        help="fraction of total set of nodes to target for sampling",
    )

    parser.add_argument(
        "--num_seed_nodes",
        default=5,
        type=int,
        help="number of nodes to compute personalized pagerank around",
    )

    parser.add_argument(
        "--infection_rate",
        default=0.1,
        type=float,
        help="infection rate for spread sampling",
    )

    parser.add_argument(
        "--removal_threshold",
        default=1,
        type=int,
        help="removal threshold for spread sampling",
    )

    args = parser.parse_args()
    return args


def check_neighborhood_intersection(v, gt_graph, S, removal_threshold):
    return len(np.intersect1d(gt_graph.get_all_neighbors(v), S)) >= removal_threshold


def main():
    args = parse_args()
    args.dataset = "MAG240M"
    print(f"Args: {args}")

    dataset = MAG240MDataset(root=os.path.join(DATASET_DIRECTORY, args.dataset))

    full_edge_list_path = os.path.join(
        DATASET_DIRECTORY, args.dataset, "full_edge_list.gt"
    )
    if not os.path.exists(full_edge_list_path):
        pp = dataset.edge_index("paper", "paper")
        pa = dataset.edge_index("author", "paper") + np.array(
            [[dataset.num_papers], [0]]
        )
        ai = dataset.edge_index("author", "institution") + np.array(
            [[dataset.num_papers], [dataset.num_papers + dataset.num_authors]]
        )

        full = np.concatenate((pp, pa, ai), axis=1, dtype=np.uint32, casting="unsafe").T
        np.savetxt("temp.csv", full, fmt="%u", delimiter=",")
        g = gt.load_graph_from_csv("temp.csv", hashed=False)
        g.save(full_edge_list_path)

    gt.openmp_set_num_threads(os.cpu_count())
    gt_graph = gt.load_graph(full_edge_list_path)

    num_nodes = dataset.num_papers + dataset.num_authors + dataset.num_institutions

    train_idx = dataset.get_idx_split("train")
    training_nodes = train_idx[np.where(dataset.paper_year[train_idx] < 2018)[0]]
    target_size = int(len(training_nodes) * args.target_frac)
    print(f"Target Size: {target_size}")

    if args.method == "ppr":
        np.random.seed(args.rng_seed)
        S = np.array([])
        nodes_per_seed = math.ceil(target_size / 10)
        while len(S) < target_size:
            seed = np.random.choice(training_nodes)
            print(f"Running Personalized PageRank around node {seed}...")
            pers = gt_graph.new_vertex_property("double")
            pers.a = np.zeros(num_nodes)
            pers.a[seed] = 1
            pr = gt.pagerank(gt_graph, pers=pers)
            S = np.union1d(
                S,
                training_nodes[
                    np.argpartition(pr.get_array()[training_nodes], -nodes_per_seed)[
                        -nodes_per_seed:
                    ]
                ],
            )

        print("Saving results to disk")
        np.save(
            os.path.join(DATASET_DIRECTORY, args.dataset, "pers_pagerank.npy"),
            S.astype(int),
        )
        print("Saved results to disk")
    elif args.method == "spread_sampling":
        if ray.is_initialized() == False:
            if "redis_password" in os.environ:
                ray.init(
                    address="auto",
                    _redis_password=os.environ["redis_password"],
                    include_dashboard=True,
                )
            else:
                ray.init(include_dashboard=True)

        print(
            f"Performing spread sampling with infection rate {args.infection_rate} and removal threshold {args.removal_threshold}..."
        )
        C = training_nodes
        S = np.array([])
        R = np.array([])
        num_splits = args.ncpus * 10
        gt_graph_ref = ray.put(gt_graph)

        while len(C) > 0 and len(S) < target_size:
            print(f"len(C)= {len(C)}, len(S)= {len(S)}")
            S = np.append(
                S,
                C[
                    np.where(np.random.uniform(0, 1, (len(C),)) < args.infection_rate)[
                        0
                    ]
                ],
            )
            C = np.setdiff1d(C, S, assume_unique=True)
            N = len(C)
            S_ref = ray.put(S)

            results_ref = [
                execute_function_on_list.remote(
                    check_neighborhood_intersection,
                    C[i * N // num_splits : (i + 1) * N // num_splits],
                    gt_graph_ref,
                    S_ref,
                    args.removal_threshold,
                )
                for i in range(num_splits)
            ]

            timeout = 30
            B_k = np.array([])
            while len(results_ref) > 0:
                ready_ref, results_ref = ray.wait(
                    results_ref, num_returns=len(results_ref), timeout=timeout
                )
                if ready_ref:
                    ready_list = ray.get(ready_ref)
                    B_k = reduce(
                        np.union1d, ([n for n, b in batch if b] for batch in ready_list)
                    )

            R = np.append(R, B_k)
            C = np.setdiff1d(C, R, assume_unique=True)
        print("Saving results to disk")
        np.save(
            os.path.join(
                DATASET_DIRECTORY,
                args.dataset,
                f"spread_sampled_p={args.infection_rate}_k={args.removal_threshold}.npy",
            ),
            S.astype(int),
        )
        print("Saved results to disk")


if __name__ == "__main__":
    main()
