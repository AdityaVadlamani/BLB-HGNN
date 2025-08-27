#!/bin/bash
#SBATCH --partition=cpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH -o ./jobs/OGB_MAG/slurm-out-%A.txt
#SBATCH -e ./jobs/OGB_MAG/slurm-err-%A.txt

# Update with your conda env name
CONDAENV=myenv

source ~/.bashrc
conda deactivate
conda activate $CONDAENV

arr = ("$@")

# __script_start__
cd $HOME/OGB_MAG
python -u memory.py --b_list $arr