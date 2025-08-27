#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial-48core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2,pfsdir
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH -o ./jobs/OGB_MAG/slurm-out-%A.txt
#SBATCH -e ./jobs/OGB_MAG/slurm-err-%A.txt

# Update with your conda env name
CONDAENV=myenv

source ~/.bashrc
conda deactivate
conda activate $CONDAENV

# __script_start__
cd $HOME/MultiBiSage
echo "Running MultiBiSage Program"
echo "-------------------------------"

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
    --n_sample_multiplier $1 \
    --num_subsets $2 \
    --use_dynamic_replica_set_frac \
    --sampler_type $3 \
    --seed $4