#!/bin/bash
#SBATCH --job-name=aortic_llm_labels
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm_llm_labels_%j.out
#SBATCH --error=results/slurm_llm_labels_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

module load anaconda
conda activate llm_extraction

cd "/home/czhan182/scr4_rsteven1/chenjia_echo_project/2026 Multi-Modal Project/scripts/echo_infer_AD"

python scripts/02_extract_labels_llm.py --config configs/default.yaml --model Qwen/Qwen2.5-7B-Instruct

echo "Job finished at: $(date)"