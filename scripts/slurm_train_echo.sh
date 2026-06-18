#!/bin/bash -l
#SBATCH --job-name=echo_baseline
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_echo_%j.out
#SBATCH --error=slurm_echo_%j.err

# ---------------------------------------------------------------------------
# Single-modality echo baseline training
#
# Usage:
#   Full run:
#     sbatch scripts/slurm_train_echo.sh
#
#   Debug (2 epochs, 50 patients, ~5 min):
#     sbatch --partition=a100 --time=00:15:00 scripts/slurm_train_echo.sh --debug
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

$PYTHON src/train_echo.py \
    --output_dir outputs/echo_baseline \
    --batch_size 16 \
    --num_epochs 50 \
    "$@"

echo "Finished: $(date)"
