#!/bin/bash
#SBATCH --job-name=panecho_finetune
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_train_%j.out
#SBATCH --error=results/slurm_train_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

module load anaconda
conda activate llm_extraction

cd "/home/czhan182/scr4_rsteven1/chenjia_echo_project/2026 Multi-Modal Project/scripts/echo_infer_AD"

python scripts/03_train.py --config configs/default.yaml

echo "Job finished at: $(date)"