#!/bin/bash -l
#SBATCH --job-name=cxr_finetune
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cxrft_%j.out
#SBATCH --error=logs/cxrft_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export HF_HOME=/scratch4/rsteven1/chenjia_echo_project/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "node $SLURMD_NODENAME gpu ${CUDA_VISIBLE_DEVICES:-none} $(date) SMOKE=${SMOKE:-0} FT_BLOCKS=${FT_BLOCKS:-2}"
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/train_cxr_finetune_episode.py
