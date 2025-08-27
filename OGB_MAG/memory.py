import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict

import _pickle as cPickle
import numpy as np
import tqdm
from constants import WORKING_DIRECTORY
from torch_geometric.datasets import OGB_MAG


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )

    parser.add_argument(
        "--b_list",
        default=[],
        type=float,
        nargs="+",
        help="list of b values to run for",
    )

    args = parser.parse_args()
    return args


def main(args):
    with open(
        os.path.join(WORKING_DIRECTORY, "datasets", "OGB_MAG/random_walkers.pickle"),
        "rb",
    ) as f:
        RWs = cPickle.load(f)

    dataset = OGB_MAG(root=os.path.join(WORKING_DIRECTORY, "datasets", "OGB_MAG"))[0]
    train_idx = dataset["paper"].train_mask.nonzero(as_tuple=True)[0].numpy()
    N = len(train_idx)

    for b in args.b_list:
        print(f"Runnning for b={b}")
        N_r = int(b * N)
        num_samples = 5 if b < 1 else 1
        avg_total_nodes = 0
        for s in range(num_samples):
            unique_nodes = defaultdict(set)
            gen = np.random.default_rng(s)
            train_nodes = train_idx[gen.choice(N, N_r, replace=False)]
            unique_nodes["paper"] = unique_nodes["paper"].union(train_nodes)
            for rw in RWs:
                for idx in tqdm.tqdm(train_nodes, mininterval=N_r // 1000):
                    neighs = rw.get_top_k_neighbors_for_node(idx, 50)
                    for neigh, _ in neighs:
                        if neigh == -1:
                            break
                        unique_nodes[rw.node_types[-1]].add(neigh)

            nodes_per_type = {k: len(v) for k, v in unique_nodes.items()}
            print(nodes_per_type)
            avg_total_nodes += sum(nodes_per_type.values()) / num_samples

    print("Average Total # of Nodes Needed for b={}: {}".format(b, avg_total_nodes))
    print(
        "Average Total Memory Needed for b = {}: {} B ".format(
            b, avg_total_nodes * 4 * 128
        )
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
