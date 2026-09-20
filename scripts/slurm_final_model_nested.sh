#!/bin/bash
#SBATCH --job-name=final_nested
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/final_nested_%j.out
#SBATCH --error=logs/final_nested_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=""
# nested = 5 outer x (5 inner + 1 refit) base fits, so ~6x the old runtime.
export FOLD_MODE="${FOLD_MODE:-immutable}" K_PCA="${K_PCA:-32}" K_ECG="${K_ECG:-32}"
export N_INNER="${N_INNER:-5}" CXR_FT="${CXR_FT:-1}" ECG="${ECG:-1}"
export OUT_DIR="${OUT_DIR:-final_model_nested}"
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/train_final_model_nested.py
