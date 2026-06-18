#!/bin/bash -l
#SBATCH --job-name=ecg_pretrain
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_pretrain_%j.out
#SBATCH --error=slurm_pretrain_%j.err

# ---------------------------------------------------------------------------
# SimCLR ECG Pretraining — SLURM GPU job
#
# Usage:
#   Full 100-epoch pretraining (batch_size=512, ~3 h on A100):
#     sbatch scripts/slurm_pretrain_ecg.sh
#
#   Debug run (2 epochs, 1000 records, ~2 min):
#     sbatch --partition=a100 --time=00:10:00 scripts/slurm_pretrain_ecg.sh --debug
#
#   Custom output path:
#     sbatch scripts/slurm_pretrain_ecg.sh --output_ckpt pretrained_checkpoints/ecg_v2.pt
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="/scratch4/rsteven1/your_env_name/bin/python3.10"
PROJECT_DIR="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
SRC_DIR="$PROJECT_DIR/src"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU(s):   $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"

cd "$PROJECT_DIR"

$PYTHON src/pretrain_ecg.py \
    --output_ckpt pretrained_checkpoints/ecg_pretrain.pt \
    --epochs 100 \
    --batch_size 512 \
    --num_workers 8 \
    "$@"

echo "Finished: $(date)"
