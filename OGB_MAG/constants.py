import os

WORKING_DIRECTORY = os.path.dirname(__file__)
SCRATCH_DIRECTORY = "/fs/scratch/PAS2030/blb"
# OUTPUT_DIRECTORY = os.path.join(SCRATCH_DIRECTORY, "output")

OUTPUT_DIRECTORY = os.path.join(WORKING_DIRECTORY, "output")
DATASET_DIRECTORY = os.path.join(WORKING_DIRECTORY, "datasets")
PREPROCESSED_GRAPHS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "preprocessed_graphs")

RANDOM_WALK_OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, "random_walks")
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, "models")
