# BLB-HGNN

This repository contains the official implementation of BLB-HGNN, a scalable training framework for heterogeneous graph neural networks (HGNNs). BLB-HGNN addresses the feature memory bottleneck in large-scale graph learning, where node features can reach terabyte or petabyte scales in real-world systems (e.g., Pinterest, Google).

**Key Idea:** BLB-HGNN uses Bag of Little Bootstraps–style sampling to train multiple HGNN replicas on fractions of the data, merge them via parameter averaging, and fine-tune a single final model.

This branch contains the latest version of the code for the BLB-HGNN introduced in the ICDM 2025 "BLB-HGNN: Bag of Little Bootstraps for Training Heterogeneous GNNs".

## Directory Structure

- `MAG240M` contains all the code for the MAG240M dataset using our own implementation of MultiBiSage [1]

- `OGB_MAG` contains all the code for the OGB-MAG dataset using our own implementation of MultiBiSage [1]

- `SeHGNN` contains all the code for the OGB-MAG dataset using SeHGNN [2]

[1] Gurukar et al., MultiBiSage: A Web-Scale Recommendation System Using Multiple Bipartite Graphs at Pinterest, VLDB 2022. [Paper](https://dl.acm.org/doi/10.14778/3574245.3574262)

[2] Yang et al., Simple and Efficient Heterogeneous Graph Neural Network, AAAI 2023. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26283/26055). [Code](https://github.com/ICT-GIMLab/SeHGNN)

## Acknowledgements
The authors acknowledge support from National Science Foundation (NSF) grants MRI OAC-2018627 and SES-1949037, and the Ohio Supercomputing Center for experimentation resources. We would also like to acknowledge the AI-Edge Institute (NSF CNS-2112471). Any opinions and findings are those of the author(s) and do not necessarily reflect the views of the granting agencies.

## Contact

<strong>Aditya Vadlamani:</strong> [Website](https://adityavadlamani.github.io/) [Email](mailto:vadlamani.12@osu.edu)
