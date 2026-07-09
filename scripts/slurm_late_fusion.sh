#!/bin/bash -l
#SBATCH --job-name=late_fusion
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/07_2026-06-18_late-fusion/slurm_late_%j.out
#SBATCH --error=logs/07_2026-06-18_late-fusion/slurm_late_%j.err

# ---------------------------------------------------------------------------
# v3 Milestone 6 — late fusion / stacking. Level-0 unimodal HGB on full
# per-modality cohorts (cross_val_predict OOF) -> level-1 LogisticRegression on
# the 522. Reports bootstrap CIs. CPU-only.
#
# Usage: sbatch scripts/slurm_late_fusion.sh
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
$PYTHON scripts/train_late_fusion.py

echo "Finished: $(date)"
