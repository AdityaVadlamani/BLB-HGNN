import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Any, Dict, Tuple

import numpy as np
import ray
import torch
from dgl import load_graphs
from torch_geometric.datasets import OGB_MAG, HGBDataset

from constants import (DATASET_DIRECTORY, PREPROCESSED_GRAPHS_DIRECTORY,
                       RANDOM_WALK_OUTPUT_PATH)
from graph_utils import (GraphDict, combine_graph_dicts,
                         edge_index_to_graph_dict, get_graph_dict,
                         save_graph_dict)
from random_walker import RandomWalker


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    parser.add_argument(
        "--dataset",
        default="OGB_MAG",
        choices=[
            "OGB_MAG",
            "academic4HetGNN",
            "ohgbn-yelp2",
            "ohgbn-Freebase",
            "IMDB",
            "DBLP",
        ],
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


@ray.remote(scheduling_strategy="SPREAD", memory=40 * 1024 * 1024 * 1024)
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

    print("Loading and preprocessing graphs...")

    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)
    if not os.path.exists(RANDOM_WALK_OUTPUT_PATH):
        os.makedirs(RANDOM_WALK_OUTPUT_PATH)
    if not os.path.exists(PREPROCESSED_GRAPHS_DIRECTORY):
        os.makedirs(PREPROCESSED_GRAPHS_DIRECTORY)

    if args.dataset == "OGB_MAG":
        dataset = OGB_MAG(os.path.join(DATASET_DIRECTORY, args.dataset))[0]
    elif args.dataset in ["academic4HetGNN", "ohgbn-yelp2", "ohgbn-Freebase"]:
        dataset = load_graphs(
            os.path.join(DATASET_DIRECTORY, args.dataset) + "/graph.bin"
        )[0][0]
    elif args.dataset in ["IMDB", "DBLP"]:
        dataset = HGBDataset(
            os.path.join(DATASET_DIRECTORY, args.dataset), args.dataset
        )[0]
    else:
        raise ValueError("Invalid dataset")

    if args.dataset in ["academic4HetGNN", "ohgbn-yelp2", "ohgbn-Freebase"]:
        node_counts: Dict[str, int] = {
            item: dataset.num_nodes(item) for item in dataset.ntypes
        }
    else:
        node_counts: Dict[str, int] = {
            str(item[0]): int(
                item[1].x.shape[0] if "x" in item[1] else int(item[1].num_nodes)
            )
            for item in dataset.node_items()
        }

    if args.dataset == "OGB_MAG":
        all_node_types = [
            ("paper", "paper"),
            ("paper", "author"),
            ("paper", "field_of_study"),
            ("paper", "institution"),
        ]

        graph_dicts = {}
        if ("paper", "paper") in all_node_types:
            paper_paper = get_graph_dict(
                dataset_name=args.dataset, node_types=["paper", "paper"]
            )
            if paper_paper is None:
                paper_paper = edge_index_to_graph_dict(
                    dataset.get_edge_store("paper", "cites", "paper")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["paper", "paper"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["paper", "paper"],
                    graph_dict=paper_paper,
                )
            graph_dicts[("paper", "paper")] = paper_paper

        if ("paper", "field_of_study") in all_node_types:
            paper_field_of_study = get_graph_dict(
                dataset_name=args.dataset, node_types=["paper", "field_of_study"]
            )
            if paper_field_of_study is None:
                paper_field_of_study = edge_index_to_graph_dict(
                    dataset.get_edge_store("paper", "has_topic", "field_of_study")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["paper", "field_of_study"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["paper", "field_of_study"],
                    graph_dict=paper_field_of_study,
                )
            graph_dicts[("paper", "field_of_study")] = paper_field_of_study

        if ("paper", "author") in all_node_types or (
            "paper",
            "institution",
        ) in all_node_types:
            paper_author = get_graph_dict(
                dataset_name=args.dataset, node_types=["paper", "author"]
            )
            if paper_author is None:
                paper_author = edge_index_to_graph_dict(
                    np.flip(
                        dataset.get_edge_store("author", "writes", "paper")[
                            "edge_index"
                        ].numpy(),
                        axis=0,
                    ),
                    node_counts=node_counts,
                    node_types=["paper", "author"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["paper", "author"],
                    graph_dict=paper_author,
                )
            graph_dicts[("paper", "author")] = paper_author

        if ("paper", "institution") in all_node_types:
            paper_institution = get_graph_dict(
                dataset_name=args.dataset, node_types=["paper", "institution"]
            )
            if paper_institution is None:
                author_institution = get_graph_dict(
                    dataset_name=args.dataset, node_types=["author", "institution"]
                )
                if author_institution is None:
                    author_institution = edge_index_to_graph_dict(
                        dataset.get_edge_store(
                            "author", "affiliated_with", "institution"
                        )["edge_index"].numpy(),
                        node_counts=node_counts,
                        node_types=["author", "institution"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
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
                    dataset_name=args.dataset,
                    node_types=["paper", "institution"],
                    graph_dict=paper_institution,
                )

                # Delete unused graph dicts
                del author_institution
                if ("paper", "author") not in all_node_types:
                    del paper_author  # pyright: ignore[reportUnboundVariable]

            graph_dicts[("paper", "institution")] = paper_institution

        starting_nodes = np.arange(node_counts["paper"])
    elif args.dataset == "academic4HetGNN":
        all_node_types = [
            ("author", "paper"),
            ("author", "venue"),
        ]

        graph_dicts = {}
        if ("author", "paper") in all_node_types or (
            "author",
            "venue",
        ) in all_node_types:
            author_paper = get_graph_dict(
                dataset_name=args.dataset, node_types=["author", "paper"]
            )
            if author_paper is None:
                author_paper = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("author", "author-paper", "paper"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["author", "paper"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["author", "paper"],
                    graph_dict=author_paper,
                )
            if ("author", "paper") in all_node_types:
                graph_dicts[("author", "paper")] = author_paper

        if ("author", "venue") in all_node_types:
            author_venue = get_graph_dict(
                dataset_name=args.dataset, node_types=["author", "venue"]
            )
            if author_venue is None:
                author_paper = get_graph_dict(
                    dataset_name=args.dataset, node_types=["author", "paper"]
                )
                paper_venue = get_graph_dict(
                    dataset_name=args.dataset, node_types=["paper", "venue"]
                )
                if paper_venue is None:
                    paper_venue = edge_index_to_graph_dict(
                        torch.stack(
                            dataset.edges(etype=("paper", "paper-venue", "venue"))
                        ).numpy(),
                        node_counts=node_counts,
                        node_types=["paper", "venue"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
                        node_types=["paper", "venue"],
                        graph_dict=paper_venue,
                    )

                author_venue = combine_graph_dicts(
                    author_paper,  # type: ignore
                    paper_venue,  # type: ignore
                    node_counts,
                    ["author", "paper", "venue"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["author", "venue"],
                    graph_dict=author_venue,
                )

                # Delete unused graph dicts
                del paper_venue
                if ("author", "paper") not in all_node_types:
                    del author_paper  # pyright: ignore[reportUnboundVariable]

            graph_dicts[("author", "venue")] = author_venue

        starting_nodes = np.arange(node_counts["author"])
    elif args.dataset == "ohgbn-yelp2":
        all_node_types = [
            ("business", "location"),
            ("business", "phrase"),
            ("business", "stars"),
        ]

        graph_dicts = {}
        if ("business", "location") in all_node_types:
            business_location = get_graph_dict(
                dataset_name=args.dataset, node_types=["business", "location"]
            )
            if business_location is None:
                business_location = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("business", "located-in", "location"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["business", "location"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["business", "location"],
                    graph_dict=business_location,
                )
            graph_dicts[("business", "location")] = business_location

        if ("business", "phrase") in all_node_types:
            business_phrase = get_graph_dict(
                dataset_name=args.dataset, node_types=["business", "phrase"]
            )
            if business_phrase is None:
                business_phrase = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("business", "described-with", "phrase"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["business", "phrase"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["business", "phrase"],
                    graph_dict=business_phrase,
                )
            graph_dicts[("business", "phrase")] = business_phrase

        if ("business", "stars") in all_node_types:
            business_stars = get_graph_dict(
                dataset_name=args.dataset, node_types=["business", "stars"]
            )
            if business_stars is None:
                business_stars = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("business", "rate", "stars"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["business", "stars"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["business", "stars"],
                    graph_dict=business_stars,
                )
            graph_dicts[("business", "stars")] = business_stars

        starting_nodes = np.arange(node_counts["business"])
    
    elif args.dataset == "ohgbn-Freebase":    
        all_node_types = [
            ("BOOK", "ORGANIZATION"),
            ("BOOK", "BOOK"),
            ("BOOK", "LOCATION"),
            ("BOOK", "SPORTS"),
            ("BOOK", "FILM"),
            ("BOOK", "BUSINESS"),
            ("BOOK", "MUSIC"),
        ]

        graph_dicts = {}
        if ("BOOK", "ORGANIZATION") in all_node_types:
            BOOK_ORGANIZATION = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "ORGANIZATION"]
            )
            if BOOK_ORGANIZATION is None:
                BOOK_ORGANIZATION = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("BOOK", "BOOK-about-ORGANIZATION", "ORGANIZATION"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["BOOK", "ORGANIZATION"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "ORGANIZATION"],
                    graph_dict=BOOK_ORGANIZATION,
                )
            if ("BOOK", "ORGANIZATION") in all_node_types:
                graph_dicts[("BOOK", "ORGANIZATION")] = BOOK_ORGANIZATION
        
        if ("BOOK", "BOOK") in all_node_types:
            BOOK_BOOK = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "BOOK"]
            )
            if BOOK_BOOK is None:
                BOOK_BOOK = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("BOOK", "BOOK-and-BOOK", "BOOK"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["BOOK", "BOOK"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "BOOK"],
                    graph_dict=BOOK_BOOK,
                )
            if ("BOOK", "BOOK") in all_node_types:
                graph_dicts[("BOOK", "BOOK")] = BOOK_BOOK

        if ("BOOK", "LOCATION") in all_node_types:
            BOOK_LOCATION = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "LOCATION"]
            )
            if BOOK_LOCATION is None:
                BOOK_LOCATION = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("BOOK", "BOOK-on-LOCATION", "LOCATION"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["BOOK", "LOCATION"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "LOCATION"],
                    graph_dict=BOOK_LOCATION,
                )
            if ("BOOK", "LOCATION") in all_node_types:
                graph_dicts[("BOOK", "LOCATION")] = BOOK_LOCATION
                
        if ("BOOK", "SPORTS") in all_node_types:
            BOOK_SPORTS = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "SPORTS"]
            )
            if BOOK_SPORTS is None:
                BOOK_SPORTS = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("BOOK", "BOOK-on-SPORTS", "SPORTS"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["BOOK", "SPORTS"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "SPORTS"],
                    graph_dict=BOOK_SPORTS,
                )
            if ("BOOK", "SPORTS") in all_node_types:
                graph_dicts[("BOOK", "SPORTS")] = BOOK_SPORTS
                
        if ("BOOK", "FILM") in all_node_types:
            BOOK_FILM = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "FILM"]
            )
            if BOOK_FILM  is None:
                BOOK_FILM  = edge_index_to_graph_dict(
                    torch.stack(
                        dataset.edges(etype=("BOOK", "BOOK-to-FILM", "FILM"))
                    ).numpy(),
                    node_counts=node_counts,
                    node_types=["BOOK", "FILM"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "FILM"],
                    graph_dict=BOOK_FILM,
                )
            if ("BOOK", "FILM") in all_node_types:
                graph_dicts[("BOOK", "FILM")] = BOOK_FILM 

        if ("BOOK", "BUSINESS") in all_node_types:
            BOOK_BUSINESS = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "BUSINESS"]
            )
            if BOOK_BUSINESS is None:
                BOOK_ORGANIZATION = get_graph_dict(
                    dataset_name=args.dataset, node_types=["BOOK", "ORGANIZATION"]
                )
                ORGANIZATION_BUSINESS = get_graph_dict(
                    dataset_name=args.dataset, node_types=["ORGANIZATION", "BUSINESS"]
                )
                if ORGANIZATION_BUSINESS is None:
                    ORGANIZATION_BUSINESS = edge_index_to_graph_dict(
                        torch.stack(
                            dataset.edges(etype=("ORGANIZATION", "ORGANIZATION-for-BUSINESS", "BUSINESS"))
                        ).numpy(),
                        node_counts=node_counts,
                        node_types=["ORGANIZATION", "BUSINESS"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
                        node_types=["ORGANIZATION", "BUSINESS"],
                        graph_dict=ORGANIZATION_BUSINESS,
                    )

                BOOK_BUSINESS = combine_graph_dicts(
                    BOOK_ORGANIZATION,  # type: ignore
                    ORGANIZATION_BUSINESS,  # type: ignore
                    node_counts,
                    ["BOOK", "ORGANIZATION", "BUSINESS"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "BUSINESS"],
                    graph_dict=BOOK_BUSINESS,
                )

                # Delete unused graph dicts
                del ORGANIZATION_BUSINESS
                if ("BOOK", "ORGANIZATION") not in all_node_types:
                    del BOOK_ORGANIZATION  # pyright: ignore[reportUnboundVariable]

            graph_dicts[("BOOK", "BUSINESS")] = BOOK_BUSINESS
            
        if ("BOOK", "MUSIC") in all_node_types:
            BOOK_MUSIC = get_graph_dict(
                dataset_name=args.dataset, node_types=["BOOK", "MUSIC"]
            )
            if BOOK_MUSIC is None:
                BOOK_ORGANIZATION = get_graph_dict(
                    dataset_name=args.dataset, node_types=["BOOK", "ORGANIZATION"]
                )
                ORGANIZATION_MUSIC = get_graph_dict(
                    dataset_name=args.dataset, node_types=["ORGANIZATION", "MUSIC"]
                )
                if ORGANIZATION_MUSIC is None:
                    ORGANIZATION_MUSIC = edge_index_to_graph_dict(
                        torch.stack(
                            dataset.edges(etype=("ORGANIZATION", "ORGANIZATION-to-MUSIC", "MUSIC"))
                        ).numpy(),
                        node_counts=node_counts,
                        node_types=["ORGANIZATION", "MUSIC"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
                        node_types=["ORGANIZATION", "MUSIC"],
                        graph_dict=ORGANIZATION_MUSIC,
                    )

                BOOK_MUSIC = combine_graph_dicts(
                    BOOK_ORGANIZATION,  # type: ignore
                    ORGANIZATION_MUSIC,  # type: ignore
                    node_counts,
                    ["BOOK", "ORGANIZATION", "MUSIC"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["BOOK", "MUSIC"],
                    graph_dict=BOOK_MUSIC,
                )

                # Delete unused graph dicts
                del ORGANIZATION_MUSIC
                if ("BOOK", "ORGANIZATION") not in all_node_types:
                    del BOOK_ORGANIZATION  # pyright: ignore[reportUnboundVariable]

            graph_dicts[("BOOK", "MUSIC")] = BOOK_MUSIC
    
        starting_nodes = np.arange(node_counts["BOOK"])
        
    elif args.dataset == "IMDB":
        all_node_types = [
            ("movie", "actor"),
            ("movie", "director"),
            ("movie", "keyword"),
        ]

        graph_dicts = {}
        if ("movie", "actor") in all_node_types:
            movie_actor = get_graph_dict(
                dataset_name=args.dataset, node_types=["movie", "actor"]
            )
            if movie_actor is None:
                movie_actor = edge_index_to_graph_dict(
                    dataset.get_edge_store("movie", ">actorh", "actor")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["movie", "actor"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["movie", "actor"],
                    graph_dict=movie_actor,
                )
            graph_dicts[("movie", "actor")] = movie_actor

        if ("movie", "director") in all_node_types:
            movie_director = get_graph_dict(
                dataset_name=args.dataset, node_types=["movie", "director"]
            )
            if movie_director is None:
                movie_director = edge_index_to_graph_dict(
                    dataset.get_edge_store("movie", "to", "director")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["movie", "director"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["movie", "director"],
                    graph_dict=movie_director,
                )
            graph_dicts[("movie", "director")] = movie_director

        if ("movie", "keyword") in all_node_types:
            movie_keyword = get_graph_dict(
                dataset_name=args.dataset, node_types=["movie", "keyword"]
            )
            if movie_keyword is None:
                movie_keyword = edge_index_to_graph_dict(
                    dataset.get_edge_store("movie", "to", "keyword")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["movie", "keyword"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["movie", "keyword"],
                    graph_dict=movie_keyword,
                )
            graph_dicts[("movie", "keyword")] = movie_keyword

        starting_nodes = np.arange(node_counts["movie"])
    elif args.dataset == "DBLP":
        all_node_types = [
            ("author", "paper"),
            ("author", "term"),
            ("author", "venue"),
        ]

        graph_dicts = {}
        if all_node_types:
            author_paper = get_graph_dict(
                dataset_name=args.dataset, node_types=["author", "paper"]
            )
            if author_paper is None:
                author_paper = edge_index_to_graph_dict(
                    dataset.get_edge_store("author", "to", "paper")[
                        "edge_index"
                    ].numpy(),
                    node_counts=node_counts,
                    node_types=["author", "paper"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["author", "paper"],
                    graph_dict=author_paper,
                )
            graph_dicts[("author", "paper")] = author_paper

        if ("author", "term") in all_node_types:
            author_term = get_graph_dict(
                dataset_name=args.dataset, node_types=["author", "term"]
            )
            if author_term is None:
                paper_term = get_graph_dict(
                    dataset_name=args.dataset, node_types=["paper", "term"]
                )
                if paper_term is None:
                    paper_term = edge_index_to_graph_dict(
                        dataset.get_edge_store("paper", "to", "term")[
                            "edge_index"
                        ].numpy(),
                        node_counts=node_counts,
                        node_types=["paper", "term"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
                        node_types=["paper", "term"],
                        graph_dict=paper_term,
                    )

                author_term = combine_graph_dicts(
                    author_paper,  # type: ignore
                    paper_term,  # type: ignore
                    node_counts,
                    ["author", "paper", "term"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["author", "term"],
                    graph_dict=author_term,
                )

                # Delete unused graph dicts
                del paper_term

            graph_dicts[("author", "term")] = author_term

        if ("author", "venue") in all_node_types:
            author_venue = get_graph_dict(
                dataset_name=args.dataset, node_types=["author", "venue"]
            )
            if author_venue is None:
                paper_venue = get_graph_dict(
                    dataset_name=args.dataset, node_types=["paper", "venue"]
                )
                if paper_venue is None:
                    paper_venue = edge_index_to_graph_dict(
                        dataset.get_edge_store("paper", "to", "venue")[
                            "edge_index"
                        ].numpy(),
                        node_counts=node_counts,
                        node_types=["paper", "venue"],
                        dtype=np.uint32,
                    )
                    save_graph_dict(
                        dataset_name=args.dataset,
                        node_types=["paper", "venue"],
                        graph_dict=paper_venue,
                    )

                author_venue = combine_graph_dicts(
                    author_paper,  # type: ignore
                    paper_venue,  # type: ignore
                    node_counts,
                    ["author", "paper", "venue"],
                    dtype=np.uint32,
                )
                save_graph_dict(
                    dataset_name=args.dataset,
                    node_types=["author", "venue"],
                    graph_dict=author_venue,
                )

                # Delete unused graph dicts
                del paper_venue

            graph_dicts[("author", "venue")] = author_venue

        if ("author", "paper") not in all_node_types:
            del author_paper  # pyright: ignore[reportUnboundVariable]
        starting_nodes = np.arange(node_counts["author"])

    print("Finished loading and preprocessing graphs")

    if ray.is_initialized() == False:
        if "redis_password" in os.environ:
            ray.init(
                address="auto",
                _redis_password=os.environ["redis_password"],
                include_dashboard=True,
            )
        else:
            ray.init(include_dashboard=True)

    print("Running random walks...")

    random_walk_config = {
        "dataset_name": args.dataset,
        "num_splits": args.nnodes * args.ncpus,
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
    ray.shutdown()


if __name__ == "__main__":
    main()
