#!/bin/bash -l
#SBATCH --job-name=gbdt_fusion
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/05_2026-06-18_gbdt/slurm_gbdt_%j.out
#SBATCH --error=logs/05_2026-06-18_gbdt/slurm_gbdt_%j.err

# ---------------------------------------------------------------------------
# v3 Milestone 3 — GBDT (HistGradientBoosting) fusion baseline on the triple
# cohort. CPU-only; runs stratified 5-fold CV + per-modality ablation + PCA.
#
# Usage:
#   sbatch scripts/slurm_gbdt_fusion.sh
#
# Prereqs (already produced):
#   pretrained_checkpoints/cohort_triple.csv     (build_triple_cohort.py)
#   pretrained_checkpoints/pclr_embeddings.pt
#   pretrained_checkpoints/raddino_embeddings.pt
#   pretrained_checkpoints/ehr_features.csv      (extract_ehr_features.py)
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
# Cap thread pools to the allocation — avoids OpenMP/BLAS over-subscription
# (a stray run on the login node spawned 158 threads and thrashed).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-8}"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/train_gbdt_fusion.py

echo "Finished: $(date)"
