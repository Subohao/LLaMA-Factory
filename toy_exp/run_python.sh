#!/bin/bash
#SBATCH --job-name=preprocess_toy_exp
#SBATCH --output=logs/preprocess_toy_exp_%j.out
#SBATCH --error=logs/preprocess_toy_exp_%j.err
#SBATCH --time=0-08:00:00
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=8000M
#SBATCH --cpus-per-task=4

# Load necessary modules (optional, if needed)
# module load anaconda3/2022.05

# Activate conda environment
source ~/.bashrc
conda activate llama_factory

# Move to your working directory
cd /ocean/projects/cis210027p/bsu5/LLaMA-Factory/toy_exp

# Set memory management flags
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42

# Run the Python script with memory optimization
python -u simu_data.py