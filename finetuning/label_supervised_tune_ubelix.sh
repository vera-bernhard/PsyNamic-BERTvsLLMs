#!/bin/bash
#SBATCH --job-name="Label-Supervised Finetune Experiments"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=logs/ls_finetune.log
#SBATCH --error=logs/ls_finetune.log

module load Anaconda3
module load CUDA/12.3.0
source activate ls_env

python -m finetuning.ls_fine_tune 