#!/bin/bash -l
#SBATCH --job-name=raddino_roi
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/08_2026-07-02_m0-m1/slurm_raddino_roi_%j.out
#SBATCH --error=logs/08_2026-07-02_m0-m1/slurm_raddino_roi_%j.err

# ---------------------------------------------------------------------------
# M4c — RAD-DINO embedding extraction on an aorta/mediastinum ROI crop (~520 pts).
# Runs the frozen ViT once per cropped image; caches 768-dim CLS embeddings.
# Runtime: a few minutes on an A100.
# Usage: sbatch scripts/slurm_extract_raddino_roi.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started:  $(date)"

cd "$PROJECT_DIR"
$PYTHON scripts/extract_raddino_roi_embeddings.py

echo "Finished: $(date)"
