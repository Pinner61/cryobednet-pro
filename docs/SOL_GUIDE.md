# SOL Supercomputing Guide

## Upload project

```bash
scp -r cryobednet-pro <ASURITE>@<SOL_LOGIN_HOST>:~/cryobednet-pro
ssh <ASURITE>@<SOL_LOGIN_HOST>
cd ~/cryobednet-pro
```

## Create environment

```bash
module avail mamba
module load mamba/latest || true
conda env create -f environment.yml
conda activate cryobednet
pip install -e .
```

## Verify GPU

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
PY
```

## Submit jobs

```bash
mkdir -p logs outputs data
python scripts/make_mock_data.py --out data/mock_tiles.npz --n 2000 --seed 42
sbatch slurm/train_gpu.sbatch
squeue -u $USER
```

## Read logs

```bash
ls logs
tail -f logs/cryobednet-train-<JOBID>.out
```

## Pull results back

```bash
scp -r <ASURITE>@<SOL_LOGIN_HOST>:~/cryobednet-pro/outputs/sol_gpu ./outputs_sol_gpu
```
