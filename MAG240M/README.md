# MultiBiSage

## Installation Steps

`pip install -r requirements.txt` in the outer directory. For our experimentation we installed CUDA 11.8 wheels of the packages (e.g., torch, dgl, etc.)

## Dataset Setup

The MAG240M dataset can be loaded from the `ogb` library. The initial load come with the features for the paper node type. We include the `author.npy` and `inst.py` from the [DGL Baseline Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M).

We also include a processed version of the metapath2vec features provided by [R-UNIMP](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2021/MAG240M/r_unimp). These are `m2v_paper.npy`, `m2v_author.npy`, and `m2v_inst.npy` which can be found [here](https://zenodo.org/records/13338650?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRhNjkxNGQwLTA4NDAtNDY4NC04ZWI0LWU5YjYwYmIyOTIxMSIsImRhdGEiOnt9LCJyYW5kb20iOiIzOWE2ZTg5NzI0MGNiYzM0NTlkNjk5YTgwOWVhYTVmZiJ9.Mk-0fDkv7yaYx1D0r1pJiZAnQb6FLmarS3DBGRnkuzo1xwtedI_m_7Swzx4su7VwGsQwSbmwge7vAKMsPe31Ag).

After downloading MAG240M, move all additional `*.npy` files to the `mag240m_kddcup2021/processed` folder.

# Running experiments

We provide a `scripts` directory with all the SLURM execution scripts executed for the jobs.
