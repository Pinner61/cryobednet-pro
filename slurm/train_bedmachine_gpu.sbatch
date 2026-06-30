#!/bin/bash
#SBATCH --job-name=cryobednet-bedmachine
#SBATCH --output=logs/cryobednet-bedmachine-train-%j.out
#SBATCH --error=logs/cryobednet-bedmachine-train-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p logs outputs
module purge
module load mamba/latest || true
source ~/.bashrc || true
conda activate cryobednet

python -m cryobednet.train --config configs/bedmachine_rrdb_gpu.yaml
