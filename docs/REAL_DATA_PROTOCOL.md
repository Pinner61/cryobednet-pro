# Real Data Protocol

This phase replaces mock tiles with Antarctic raster tiles from BedMachine Antarctica.

## Scientific design

The first real-data benchmark is controlled super-resolution over BedMachine Antarctica:

- `y_hr`: original BedMachine bed elevation tile
- `x_lr`: deliberately downsampled bed tile
- `a_hr`: surface elevation, ice thickness, and mask
- `region`: spatial block label for leakage-aware holdout testing

This is not the final scientific claim. It is the first reproducible real-data benchmark. The stronger follow-up is cross-source residual learning using independent inputs such as BEDMAP2, REMA, MEaSUREs velocity, and radar-derived observations.

## Build tiles

```bash
BM=$(ls data/raw/*BedMachine*Antarctica*.nc | head -1)
python scripts/build_bedmachine_tiles.py \
  --nc "$BM" \
  --out data/bedmachine_tiles.npz \
  --tile-hr 64 \
  --scale 4 \
  --stride 64 \
  --max-tiles 3000 \
  --seed 42
```

Print region counts:

```bash
python scripts/print_region_counts.py data/bedmachine_tiles.npz
```

Set `split.holdout_region` in `configs/bedmachine_rrdb_gpu.yaml` to a block with enough tiles.

## Train and evaluate on SOL

```bash
sbatch slurm/train_bedmachine_gpu.sbatch
squeue -u $USER
```

After training:

```bash
sbatch slurm/evaluate_bedmachine_cpu.sbatch
```

Copy selected figures to `reports/real_bedmachine/` before committing. Do not commit raw data, `.nc` files, `.npz` tiles, or checkpoints.
