from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr
from skimage.transform import resize


ALIASES = {
    "bed": ["bed", "bed_elevation", "bedrock", "topg"],
    "surface": ["surface", "surface_elevation", "usrf", "dem"],
    "thickness": ["thickness", "thick", "ice_thickness"],
    "mask": ["mask", "bedmachine_mask", "icemask"],
}


def pick_var(ds: xr.Dataset, key: str) -> str:
    lower = {name.lower(): name for name in ds.data_vars}
    for alias in ALIASES[key]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise KeyError(f"Could not find {key}. Available variables: {list(ds.data_vars)}")


def finite_fraction(*arrays: np.ndarray) -> float:
    valid = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        valid &= np.isfinite(arr)
    return float(valid.mean())


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((arr - mean) / (std + 1e-6)).astype("float32")


def make_lowres(arr: np.ndarray, lr_size: int) -> np.ndarray:
    return resize(arr, (lr_size, lr_size), order=1, mode="reflect", anti_aliasing=True).astype("float32")


def block_labels(xs: np.ndarray, ys: np.ndarray, n: int) -> np.ndarray:
    x_edges = np.quantile(xs, np.linspace(0, 1, n + 1))
    y_edges = np.quantile(ys, np.linspace(0, 1, n + 1))
    x_bin = np.clip(np.digitize(xs, x_edges[1:-1]), 0, n - 1)
    y_bin = np.clip(np.digitize(ys, y_edges[1:-1]), 0, n - 1)
    return np.asarray([f"block_{yb}_{xb}" for xb, yb in zip(x_bin, y_bin)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True)
    parser.add_argument("--out", default="data/bedmachine_tiles.npz")
    parser.add_argument("--tile-hr", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-tiles", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid", type=float, default=0.95)
    parser.add_argument("--block-grid", type=int, default=5)
    parser.add_argument("--valid-mask-codes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    nc = Path(args.nc)
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    ds = xr.open_dataset(nc, decode_cf=False)
    bed_name = pick_var(ds, "bed")
    surface_name = pick_var(ds, "surface")
    thickness_name = pick_var(ds, "thickness")
    mask_name = pick_var(ds, "mask")

    bed_da = ds[bed_name]
    surface_da = ds[surface_name]
    thickness_da = ds[thickness_name]
    mask_da = ds[mask_name]

    y_dim, x_dim = bed_da.dims[-2], bed_da.dims[-1]
    ny, nx = bed_da.shape[-2], bed_da.shape[-1]
    tile_hr = args.tile_hr
    lr_size = tile_hr // args.scale

    candidates = [(iy, ix) for iy in range(0, ny - tile_hr, args.stride) for ix in range(0, nx - tile_hr, args.stride)]
    rng.shuffle(candidates)

    beds, surfaces, thicknesses, masks, centers_x, centers_y = [], [], [], [], [], []

    x_coord = ds[x_dim].values if x_dim in ds.coords else np.arange(nx)
    y_coord = ds[y_dim].values if y_dim in ds.coords else np.arange(ny)

    for iy, ix in candidates:
        sl = {y_dim: slice(iy, iy + tile_hr), x_dim: slice(ix, ix + tile_hr)}
        bed = np.asarray(bed_da.isel(**sl).values, dtype="float32")
        surface = np.asarray(surface_da.isel(**sl).values, dtype="float32")
        thick = np.asarray(thickness_da.isel(**sl).values, dtype="float32")
        mask = np.asarray(mask_da.isel(**sl).values, dtype="float32")

        if args.valid_mask_codes:
            mask_ok = np.isin(mask, args.valid_mask_codes)
            if float(mask_ok.mean()) < args.min_valid:
                continue

        if finite_fraction(bed, surface, thick, mask) < args.min_valid:
            continue

        beds.append(bed)
        surfaces.append(surface)
        thicknesses.append(thick)
        masks.append(mask)
        centers_x.append(float(x_coord[ix + tile_hr // 2]))
        centers_y.append(float(y_coord[iy + tile_hr // 2]))

        if len(beds) >= args.max_tiles:
            break

    if not beds:
        raise RuntimeError("No valid tiles were produced. Try reducing --min-valid or changing --valid-mask-codes.")

    y_raw = np.stack(beds).astype("float32")
    surface_raw = np.stack(surfaces).astype("float32")
    thickness_raw = np.stack(thicknesses).astype("float32")
    mask_raw = np.stack(masks).astype("float32")
    centers_x = np.asarray(centers_x, dtype="float32")
    centers_y = np.asarray(centers_y, dtype="float32")

    bed_mean, bed_std = float(np.nanmean(y_raw)), float(np.nanstd(y_raw))
    surface_mean, surface_std = float(np.nanmean(surface_raw)), float(np.nanstd(surface_raw))
    thick_mean, thick_std = float(np.nanmean(thickness_raw)), float(np.nanstd(thickness_raw))

    y_hr = zscore(y_raw, bed_mean, bed_std)[:, None]
    x_lr = np.stack([make_lowres(tile, lr_size) for tile in y_raw])
    x_lr = zscore(x_lr, bed_mean, bed_std)[:, None]

    surface_z = zscore(surface_raw, surface_mean, surface_std)
    thick_z = zscore(thickness_raw, thick_mean, thick_std)
    mask_norm = (mask_raw > 0).astype("float32")
    a_hr = np.stack([surface_z, thick_z, mask_norm], axis=1).astype("float32")

    region = block_labels(centers_x, centers_y, args.block_grid)
    tile_id = np.asarray([f"{region[i]}_{i:06d}" for i in range(len(region))])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_lr=x_lr,
        a_hr=a_hr,
        y_hr=y_hr,
        region=region,
        tile_id=tile_id,
        center_x=centers_x,
        center_y=centers_y,
    )

    counts = Counter(region.tolist())
    summary = {
        "source": str(nc),
        "out": str(out),
        "n_tiles": int(len(region)),
        "tile_hr": tile_hr,
        "scale": args.scale,
        "lowres_tile": lr_size,
        "variables": {
            "bed": bed_name,
            "surface": surface_name,
            "thickness": thickness_name,
            "mask": mask_name,
        },
        "normalization": {
            "bed_mean": bed_mean,
            "bed_std": bed_std,
            "surface_mean": surface_mean,
            "surface_std": surface_std,
            "thickness_mean": thick_mean,
            "thickness_std": thick_std,
        },
        "region_counts": dict(sorted(counts.items())),
        "recommended_holdout": counts.most_common(1)[0][0],
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
