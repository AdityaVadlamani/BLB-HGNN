#!/bin/bash
#SBATCH --account general
#SBATCH --partition cpu_batch
#SBATCH --cpus-per-task=1
#SBATCH -o ./jobs/slurm-out-%A.txt
#SBATCH -e ./jobs/slurm-err-%A.txt

source ~/.bashrc
conda deactivate
conda activate myenv

arr = ("$@")
cd $HOME/MAG240M
python3 memory.py --b_list $arr