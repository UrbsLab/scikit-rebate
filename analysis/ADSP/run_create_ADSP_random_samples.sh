#!/bin/bash
#SBATCH --job-name=create_ADSP_random_samples
#SBATCH --output=logs/create_ADSP_random_samples_%A.out
#SBATCH --error=logs/create_ADSP_random_samples_%A.err
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=30-00:00:00
#SBATCH --partition=defq

source /home/kazemik/miniconda3/etc/profile.d/conda.sh
conda activate rebate_env
python create_ADSP_random_samples.py