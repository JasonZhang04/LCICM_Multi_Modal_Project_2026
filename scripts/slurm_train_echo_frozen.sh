#!/bin/bash -l
#SBATCH --job-name=echo_frozen
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_echo_frozen_%j.out
#SBATCH --error=slurm_echo_frozen_%j.err

# ---------------------------------------------------------------------------
# Train a small MLP head on pre-extracted PanEcho embeddings.
# Must run extract_panecho_embeddings.py (Step 1) first.
#
# Runtime: < 10 minutes (no DICOM I/O, pure CPU/GPU tensor ops).
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

$PYTHON src/train_echo_frozen.py \
    --output_dir outputs/echo_frozen \
    --num_epochs 200 \
    "$@"

echo "Finished: $(date)"
