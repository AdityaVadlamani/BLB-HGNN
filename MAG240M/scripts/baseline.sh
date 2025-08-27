#!/bin/bash
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=3-00:00:00
#SBATCH -o ./jobs/MAG240M/slurm-out-%A.txt
#SBATCH -e ./jobs/MAG240M/slurm-err-%A.txt

# Update with your conda env name
CONDAENV=myenv

source ~/.bashrc
conda deactivate
conda activate $CONDAENV

# __script_start__
cd $HOME/MAG240M
echo "Running MultiBiSage Program"
echo "-------------------------------"

srun python -u main.py \
    --job_id $SLURM_JOB_ID \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --num_gpus $SLURM_NTASKS_PER_NODE \
    --num_workers $SLURM_CPUS_PER_TASK \
    --epochs 100 \
    --dropout 0.7 \
    --batch_size 256 \
    --embedding_dim 512 \
    --num_heads 8 \
    --num_layers 2 \
    --n_sample_multiplier $1 \
    --seed $2 \
    --use_dynamic_replica_set_frac \
    --construct_new_val_set \
    --no_bootstrap \
    --no_ensemble \
