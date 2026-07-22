#!/bin/bash -l
#SBATCH --job-name=combined_stack
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=24G --time=00:50:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
PY="/scratch4/rsteven1/your_env_name/bin/python3.10"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
echo "### BEST: frontal + anatomy-ROI ###"
VIEW=frontal CXR_EMB=raddino_multi_anatomy_embeddings.pt OUT_TAG=combined_stack_frontal_anat $PY -u scripts/train_combined_stack.py
echo "### ablation: frontal + whole-image ###"
VIEW=frontal CXR_EMB=raddino_multi_embeddings.pt          OUT_TAG=combined_stack_frontal_whole $PY -u scripts/train_combined_stack.py
