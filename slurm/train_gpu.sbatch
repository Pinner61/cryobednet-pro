#!/bin/bash
#SBATCH --job-name=cryobednet-train
#SBATCH --output=logs/cryobednet-train-%j.out
#SBATCH --error=logs/cryobednet-train-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
mkdir -p logs outputs
module purge
module load mamba/latest || true
source ~/.bashrc || true
conda activate cryobednet
python -m cryobednet.train --config configs/sol_gpu.yaml
