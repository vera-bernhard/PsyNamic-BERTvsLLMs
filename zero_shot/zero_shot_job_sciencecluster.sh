#!/bin/bash
#SBATCH --job-name="Instruction Tuning Experiments"
#SBATCH --time=01:00:00
#SBATCH --gpus=A100:2
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=logs/zero_shot_mellama70B.log
#SBATCH --error=logs/zero_shot_mellama70B.log

module load mamba
module load multigpu
module load cuda/12.6.2
source activate final_env

python -m zero_shot.predict_zero_shot
