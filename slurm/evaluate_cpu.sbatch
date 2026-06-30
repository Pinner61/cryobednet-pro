#!/bin/bash
#SBATCH --job-name=cryobednet-eval
#SBATCH --output=logs/cryobednet-eval-%j.out
#SBATCH --error=logs/cryobednet-eval-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs
module purge
module load mamba/latest || true
source ~/.bashrc || true
conda activate cryobednet
python -m cryobednet.evaluate --config configs/sol_gpu.yaml --split holdout
python scripts/export_figures.py --run outputs/sol_gpu --split holdout
