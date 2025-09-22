#!/bin/bash
#SBATCH --job-name="Zero Shot Experiments"
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --gpus=A100:1
#SBATCH --output=logs/fine_tune.log
#SBATCH --error=logs/fine_tune.log

module load mamba
module load gpu
module load cuda/12.6.2
source activate final_env

python -m finetuning.instruction_tune