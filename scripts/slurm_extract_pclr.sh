#!/bin/bash -l
#SBATCH --job-name=pclr_extract
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_pclr_extract_%j.out
#SBATCH --error=slurm_pclr_extract_%j.err

# ---------------------------------------------------------------------------
# One-time PCLR embedding extraction for the aortic cohort (~2,902 patients).
# Runs TF on CPU — no GPU needed, uses the 'shared' (CPU) partition.
# Runtime: ~3-5 minutes.
#
# Usage:
#   sbatch scripts/slurm_extract_pclr.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Keep TF on CPU even if a GPU happens to be visible
export CUDA_VISIBLE_DEVICES=""

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/extract_pclr_embeddings.py

echo "Finished: $(date)"
