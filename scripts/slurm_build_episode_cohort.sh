#!/bin/bash
#SBATCH --job-name=build_episode_cohort
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/build_cohort_%j.out
#SBATCH --error=logs/build_cohort_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/build_episode_cohort.py "$@"
