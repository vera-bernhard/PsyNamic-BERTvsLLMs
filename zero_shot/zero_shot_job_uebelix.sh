#!/bin/bash
#SBATCH --job-name="Zero Shot Experiments"
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=logs/zero_shot.log
#SBATCH --error=logs/zero_shot.log

module load Anaconda3
module load CUDA/12.3.0
source activate ma_env

python -m zero_shot.predict_zero_shot