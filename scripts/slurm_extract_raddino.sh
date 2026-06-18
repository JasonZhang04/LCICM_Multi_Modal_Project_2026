#!/bin/bash -l
#SBATCH --job-name=raddino_extract
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_raddino_extract_%j.out
#SBATCH --error=slurm_raddino_extract_%j.err

# ---------------------------------------------------------------------------
# One-time RAD-DINO embedding extraction for the aortic cohort CXRs (~520 pts).
# Runs the frozen ViT once per patient image and caches 768-dim CLS embeddings.
# Runtime: a few minutes on an A100.
#
# Usage:
#   sbatch scripts/slurm_extract_raddino.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Use cached HuggingFace weights — avoids 429 rate-limit delays on startup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/extract_raddino_embeddings.py

echo "Finished: $(date)"
