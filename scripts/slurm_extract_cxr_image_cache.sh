#!/bin/bash
#SBATCH --job-name=cxr_image_cache
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/imgcache_%j.out
#SBATCH --error=logs/imgcache_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=""
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/extract_cxr_image_cache.py
