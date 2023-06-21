#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/home/yserrano/data/%x-%j.out
#SBATCH --error=/home/yserrano/data/%x-%j.err
#SBATCH --time=05-23:00:00
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH	--mem=150G
#SBATCH --exclude=node001,gpu002

ml Miniconda3 CUDA/11.6.0
source activate transformer

srun python -u inference.py 29 1000 "weights/MDH-PLM/epoch_15.ckpt"
