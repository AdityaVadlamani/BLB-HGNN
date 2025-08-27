#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH -o ./jobs/slurm-out-%A.txt
#SBATCH -e ./jobs/slurm-err-%A.txt


# Update with your conda env name
CONDAENV=myenv

source ~/.bashrc
conda deactivate
conda activate $CONDAENV

# __script_start__
cd $HOME/OGB_MAG
echo "Running MultiBiSage Program"
echo "-------------------------------"

args=("$@")

srun python -u main.py \
    --job_id $SLURM_JOB_ID \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --num_gpus $SLURM_NTASKS_PER_NODE \
    --num_workers $SLURM_CPUS_PER_TASK \
    --dataset OGB_MAG \
    --epochs 50 \
    --ensemble_epochs 50 \
    --dropout 0.6 \
    --batch_size 256 \
    --embedding_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --n_sample_multiplier ${args[0]} \
    --num_subsets ${args[1]} \
    --use_dynamic_replica_set_frac \
    --skip_replicas \
    --seed  ${args[2]} \
    --ensemble_method  ${args[3]} \
    --resume_from ${args[@]:4}