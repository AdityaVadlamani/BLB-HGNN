# MultiBiSage

## Installation Steps

`pip install -r requirements.txt` in the outer directory. For our experimentation we installed CUDA 11.8 wheels of the packages (e.g., torch, dgl, etc.)

## Dataset Setup

The OGB_MAG dataset is loaded from pytorch_geometric. The constructor is called with `preprocess='metapath2vec'`

# Running experiments

We provide a `scripts` directory with all the SLURM execution scripts executed for the jobs.
