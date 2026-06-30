#!/bin/bash
#SBATCH --job-name=cryobednet-terrain-eval
#SBATCH --output=logs/cryobednet-terrain-eval-%j.out
#SBATCH --error=logs/cryobednet-terrain-eval-%j.err
#SBATCH --partition=public
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -eo pipefail

cd ~/cryobednet-pro

mkdir -p logs outputs

module purge
module load mamba/latest

source activate cryobednet

python -m cryobednet.evaluate --config configs/bedmachine_terrain_gpu.yaml --split holdout
python scripts/export_figures.py --run outputs/bedmachine_terrain_gpu --split holdout
