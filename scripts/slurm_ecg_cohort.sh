#!/bin/bash
#SBATCH --job-name=ecg_wave_cohort
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/ecgcoh_%j.out
#SBATCH --error=logs/ecgcoh_%j.err
set -euo pipefail
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"; mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=""
/scratch4/rsteven1/your_env_name/bin/python3.10 scripts/build_ecg_waveform_cohort.py
