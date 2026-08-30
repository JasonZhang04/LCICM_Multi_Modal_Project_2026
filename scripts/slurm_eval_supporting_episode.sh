#!/bin/bash
#SBATCH --job-name=eval_supporting
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/evalsup_%j.out
#SBATCH --error=logs/evalsup_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=""
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/eval_supporting_episode.py
