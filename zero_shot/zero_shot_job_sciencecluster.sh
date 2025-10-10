#!/bin/bash
#SBATCH --job-name="Instruction Tuning Experiments"
#SBATCH --time=4:00:00
#SBATCH --gpus=A100:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=logs/zero_few_shot_medgemma.log
#SBATCH --error=logs/zero_few_shot_medgemma.log

module load mamba
module load gpu
module load cuda/12.6.2
source activate final_env

python -m zero_shot.predict_zero_shot
