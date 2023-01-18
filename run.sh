#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

module load Miniconda3/4.9.2

conda env create -f environment.yml 
source activate transformer

module load CUDA/11.6.0 Python/3.9.6-GCCcore-11.2.0 PyTorch/1.12.1-CUDA-11.6 matplotlib/3.4.3-intel-2021b

python3 main.py
