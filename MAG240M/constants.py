import os

WORKING_DIRECTORY = os.path.dirname(__file__)

# Paths to the data
DATASET_DIRECTORY = os.path.join(WORKING_DIRECTORY, "datasets")
PREPROCESSED_GRAPHS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "preprocessed_graphs")

MAG240M_PAPER_LABELS = os.path.join(
    DATASET_DIRECTORY,
    "MAG240M",
    "mag240m_kddcup2021",
    "processed",
    "paper",
    "node_label.npy",
)

MAG240M_PAPER_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "paper.npy"
)

MAG240M_AUTHOR_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "author.npy"
)

MAG240M_INST_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "inst.npy"
)

MAG240M_M2V_PAPER_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "m2v_paper.npy"
)

MAG240M_M2V_AUTHOR_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "m2v_author.npy"
)

MAG240M_M2V_INST_FEATURES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_kddcup2021", "processed", "m2v_inst.npy"
)

MAG240M_PAGERANK_VALUES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_pagerank.npy"
)

MAG240M_BETWEENNESS_VALUES = os.path.join(
    DATASET_DIRECTORY, "MAG240M", "mag240m_arxiv_betweenness.npy"
)

# Paths to the output
OUTPUT_DIRECTORY = os.path.join(WORKING_DIRECTORY, "output")
RANDOM_WALK_OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, "random_walks")
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, "models")
SUBMISSION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "submissions")
