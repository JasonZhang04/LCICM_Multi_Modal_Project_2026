#!/bin/bash -l
#SBATCH --job-name=extract_multi_cxr
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --gres=gpu:1 --mem=32G --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
PY="/scratch4/rsteven1/your_env_name/bin/python3.10"
export HF_HOME=/scratch4/rsteven1/chenjia_echo_project/hf_home
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
$PY -u scripts/extract_multi_cxr_instances.py
