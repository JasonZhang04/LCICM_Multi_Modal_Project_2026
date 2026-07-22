#!/bin/bash -l
#SBATCH --job-name=geometry_stack
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=24G --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
PY="/scratch4/rsteven1/your_env_name/bin/python3.10"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
echo "Node $SLURMD_NODENAME $(date)"
SEEDS=1,2,3,4,5 $PY -u scripts/train_geometry_stack.py
echo "Finished: $(date)"
