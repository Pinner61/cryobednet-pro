#!/bin/bash
#SBATCH --job-name=cryobednet-terrain
#SBATCH --output=logs/cryobednet-terrain-train-%j.out
#SBATCH --error=logs/cryobednet-terrain-train-%j.err
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -eo pipefail

cd ~/cryobednet-pro

mkdir -p logs outputs

module purge
module load mamba/latest

source activate cryobednet

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -m cryobednet.train --config configs/bedmachine_terrain_gpu.yaml
