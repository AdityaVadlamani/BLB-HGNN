#!/bin/bash
#SBATCH --account general
#SBATCH --partition=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o ./jobs/slurm-out-%A.txt
#SBATCH -e ./jobs/slurm-err-%A.txt

# Update with your conda env name
CONDAENV=myenv

source ~/.bashrc
conda deactivate
conda activate $CONDAENV

export DGLBACKEND=pytorch

# __script_start__
cd $HOME/SeHGNN/
echo "Running SeHGNN Program"
echo "-------------------------------"

if [ $# -eq 2 ]
then
    echo python -u blb_wrapper.py --b $1 --use_blb --s $3 --use_dynamic_replica_set_frac --dataset ogbn-mag
    python -u blb_wrapper.py --b $1 --use_blb --s $3 --use_dynamic_replica_set_frac --dataset ogbn-mag
elif [ $# -eq 1 ]
then
    echo python -u blb_wrapper.py --b $1 --dataset ogbn-mag
    python -u blb_wrapper.py --b $1 --dataset ogbn-mag
fi