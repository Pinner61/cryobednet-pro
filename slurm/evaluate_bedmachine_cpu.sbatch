#!/bin/bash
#SBATCH --job-name=cryobednet-bedmachine-eval
#SBATCH --output=logs/cryobednet-bedmachine-eval-%j.out
#SBATCH --error=logs/cryobednet-bedmachine-eval-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p logs outputs
module purge
module load mamba/latest || true
source ~/.bashrc || true
conda activate cryobednet

python -m cryobednet.evaluate --config configs/bedmachine_rrdb_gpu.yaml --split holdout
python scripts/export_figures.py --run outputs/bedmachine_rrdb_gpu --split holdout
