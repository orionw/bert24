#!/bin/bash
#SBATCH --job-name=bert24_sweep
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --exclude=c001,h01,h10,h06,h09
#SBATCH --cpus-per-task=50
#SBATCH --mem=512G
#SBATCH --output=/scratch/bvandur1/oweller2/retrieval_pretraining/logs/sweep/mini/%x_%j.log
#SBATCH --error=/scratch/bvandur1/oweller2/retrieval_pretraining/logs/sweep/mini/%x_%j.log

source /home/oweller2/miniconda3/etc/profile.d/conda.sh
conda activate bert24
cd /home/oweller2/my_scratch/retrieval_pretraining/bert24
python sweep_agent.py
