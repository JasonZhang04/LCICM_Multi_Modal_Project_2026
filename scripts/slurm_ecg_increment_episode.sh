#!/bin/bash
#SBATCH --job-name=ecg_increment_episode
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ecg_incr_%j.out
#SBATCH --error=logs/ecg_incr_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="" MOCK=0
export SEEDS="${SEEDS:-1,2,3,4,5}"
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/train_ecg_increment_episode.py
