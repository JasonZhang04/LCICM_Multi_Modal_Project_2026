#!/bin/bash -l
#SBATCH --job-name=extract_multi_anat_roi
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --gres=gpu:1 --mem=32G --time=00:50:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
# roi_venv inherits the base env (torch/transformers) and adds torchxrayvision (segmentation)
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/scratch4/rsteven1/chenjia_echo_project/hf_home
export XDG_CACHE_HOME=/scratch4/rsteven1/chenjia_echo_project/.cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
VENV=/scratch4/rsteven1/chenjia_echo_project/roi_venv
cd "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
$VENV/bin/python -u scripts/extract_multi_anatomy_roi_embeddings.py
