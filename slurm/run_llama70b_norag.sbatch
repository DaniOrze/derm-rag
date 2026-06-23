#!/bin/bash
#SBATCH --job-name=llama70b-norag
#SBATCH --output=logs/llama70b_norag_%j.out
#SBATCH --error=logs/llama70b_norag_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# Submissão:
#     sbatch slurm/run_llama70b_norag.sbatch

echo "Job $SLURM_JOB_ID em $SLURMD_NODENAME — início $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Cache do HuggingFace na pasta privada (evita encher o container de 30GB)
export HF_HOME=/home/jovyan/privado/.cache/huggingface
mkdir -p "$HF_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate derm-rag-retrieval

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs results

python scripts/run_rag_generation.py \
    --config configs/rag_generation_llama70b_norag_2gpu.yaml

echo "Fim $(date)"
