#!/bin/bash
#SBATCH --job-name=derm-retrieval
#SBATCH --output=logs/retrieval_%j.out
#SBATCH --error=logs/retrieval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ---------------------------------------------------------------------
# Experimento de retrieval denso com BGE-M3 no HPC da UFSC.
#
# Submissão:
#     sbatch slurm/run_retrieval.sbatch
#
# AJUSTE conforme o cluster: --partition, --account, módulos a carregar.
# ---------------------------------------------------------------------

echo "Job $SLURM_JOB_ID em $SLURMD_NODENAME — início $(date)"

# Carrega módulos (ajuste ao seu HPC)
# module load miniconda/3
# module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate derm-rag-retrieval

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs results

# Mostra a GPU disponível
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"

# Roda o experimento
python scripts/run_retrieval.py --config configs/dense_baseline.yaml

echo "Fim $(date)"
