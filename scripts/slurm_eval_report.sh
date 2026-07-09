#!/bin/bash -l
#SBATCH --job-name=eval_report
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/08_2026-07-02_m0-m1/slurm_evalrep_%j.out
#SBATCH --error=logs/08_2026-07-02_m0-m1/slurm_evalrep_%j.err

# ---------------------------------------------------------------------------
# M2 — unified evaluation report over all standardized OOF predictions.
# Calibration + clinical utility + PAIRED deltas vs the EHR floor. CPU-only.
# Prereq: each model's outputs/<model>/oof_predictions.csv (M0/M1 runs).
# Usage:  sbatch scripts/slurm_eval_report.sh
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
$PYTHON scripts/make_eval_report.py

echo "Finished: $(date)"
