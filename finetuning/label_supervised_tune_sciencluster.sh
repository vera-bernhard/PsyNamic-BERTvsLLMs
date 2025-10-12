#!/bin/bash
#SBATCH --job-name="Label-Supervised Finetune Experiments"
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --gpus=A100:1
#SBATCH --output=logs/ls_fine_tune.log
#SBATCH --error=logs/ls_fine_tune.log

module load mamba
module load gpu
module load cuda/12.6.2
source activate ls_env

python -m finetuning.ls_fine_tune 