#!/bin/bash

# Example slurm script to do inference with Llama-3-8B model on Yen GPU

#SBATCH -J llama
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:00:00                 # limit of 1 day runtime
#SBATCH -G 1                       # limit of 2 GPUs per user
#SBATCH -C "GPU_MODEL:A40"         # Constraint bigger GPU RAM -- 48 G
#SBATCH --mem=40G                  # CPU RAM 
#SBATCH -o llama-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@stanford.edu

# Activate venv 
source env/bin/activate

# Set the HF_HOME environment variable based on the current user
# Avoid using your home directory on the Yens due to large model size 
export HF_HOME="/scratch/shared/$USER/.cache/huggingface/hub"

# Run python script on GPU
python documented_code_example.py 
