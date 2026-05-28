#!/bin/bash -l
#SBATCH --job-name=aorta_train
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ---------------------------------------------------------------------------
# Multimodal Aortic Diameter Prediction — SLURM GPU training job
#
# Usage:
#   Full training:
#     sbatch slurm_train.sh
#
#   Debug run (2 epochs, batch_size=4 — for pipeline verification):
#     sbatch --partition=a100 --time=00:30:00 slurm_train.sh --debug
#
#   Resume from checkpoint:
#     sbatch slurm_train.sh --resume /path/to/best_model.pt
# ---------------------------------------------------------------------------

set -euo pipefail

# No module loading needed — PyTorch 2.5.1 bundles its own CUDA runtime
# (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, etc. installed in the venv).
# The GPU node provides the NVIDIA driver; the venv provides everything else.

PYTHON=/scratch4/rsteven1/your_env_name/bin/python3.10
SRC_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/src"

# Use cached HuggingFace models — avoids 429 rate-limit delays on startup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU(s):   $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"

cd "$SRC_DIR"

# Run training (any extra args passed to sbatch are forwarded here)
$PYTHON run_training.py "$@"

echo "Finished: $(date)"
