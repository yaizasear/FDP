#!/bin/bash
#SBATCH --job-name=pre-wot
#SBATCH --output=/home/yserrano/data/%x-%j.out
#SBATCH --error=/home/yserrano/data/%x-%j.err
#SBATCH --time=05-23:00:00
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=30
#SBATCH	--mem=500G
#SBATCH --exclude=node001,gpu002

ml Miniconda3
source activate transformer

cd /home/yserrano/transformer/src-tags
wandb login fbcf821ca3c41d77d2b07a61d25818130195bb27

srun python -u main.py $SLURM_JOB_ID
