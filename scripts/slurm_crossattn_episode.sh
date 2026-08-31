#!/bin/bash -l
#SBATCH --job-name=crossattn_episode
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1 --nodes=1 --cpus-per-task=6 --mem=32G --time=02:00:00
#SBATCH --output=logs/crossattn_%j.out
#SBATCH --error=logs/crossattn_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SEEDS="${SEEDS:-1,2,3}"
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/train_crossattn_episode.py
