#!/bin/bash
#SBATCH --job-name=extract_ehr_episode
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/extract_ehr_%j.out
#SBATCH --error=logs/extract_ehr_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/extract_ehr_features.py --level "${1:-episode}"
