#!/bin/bash -l
#SBATCH --job-name=ehr_baseline
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/08_2026-07-02_m0-m1/slurm_ehr_%j.out
#SBATCH --error=logs/08_2026-07-02_m0-m1/slurm_ehr_%j.err

# ---------------------------------------------------------------------------
# M1 — EHR-only clinical baseline (the lower-bound "floor"). CPU-only.
# Trains EHR HGB/LogReg/Ridge on the FULL EHR cohort via leakage-safe
# cross_val_predict, reports OOF on the n=522 triple cohort with the immutable
# fold assignments. Anchor: root >=4.0 HGB AUROC should reproduce ~0.78.
#
# Prereq: pretrained_checkpoints/fold_assignments.csv (build_fold_assignments.py)
# Usage:  sbatch scripts/slurm_ehr_baseline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
# Cap thread pools to the allocation — avoids OpenMP/BLAS over-subscription.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-8}"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/train_ehr_baseline.py

echo "Finished: $(date)"
