#!/bin/bash -l
#SBATCH --job-name=panecho_extract
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_panecho_extract_%j.out
#SBATCH --error=slurm_panecho_extract_%j.err

# ---------------------------------------------------------------------------
# One-time PanEcho embedding extraction for all aortic cohort patients.
#
# Runs PanEcho inference once offline, saves:
#   pretrained_checkpoints/panecho_embeddings.pt  ->  {subject_id: tensor(768,)}
#
# Runtime: ~3-6 hours (bottleneck is DICOM I/O, not GPU).
# Follow-up: sbatch scripts/slurm_train_echo_frozen.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="/scratch4/rsteven1/torch_hub"
export PYTHONHTTPSVERIFY=0
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU(s):   $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"

cd "$PROJECT_DIR"

$PYTHON scripts/extract_panecho_embeddings.py

echo "Finished: $(date)"
