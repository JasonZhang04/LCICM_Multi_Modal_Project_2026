#!/bin/bash -l
#SBATCH --job-name=residual_fusion
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/08_2026-07-02_m0-m1/slurm_resid_%j.out
#SBATCH --error=logs/08_2026-07-02_m0-m1/slurm_resid_%j.err

# ---------------------------------------------------------------------------
# M3 — residualized & body-size-indexed diameter modeling. CPU-only.
# Tests whether ECG+CXR add aortic-size signal beyond body size (paired delta of
# residual-fusion minus clinical-only on the same held-out patients).
# Prereq: pretrained_checkpoints/fold_assignments.csv (build_fold_assignments.py)
# Usage:  sbatch scripts/slurm_residual_fusion.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/train_residual_fusion.py

echo "Finished: $(date)"
