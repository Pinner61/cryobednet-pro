from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from skimage.transform import resize

REGIONS = np.array(["pine_island", "thwaites", "rutford", "wilkes", "ross"])


def terrain(size, rng, roughness):
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    base = -0.8 * np.exp(-3.0 * (x**2 + 0.5 * y**2))
    waves = 0.25 * np.sin(roughness * x + 0.8 * y) + 0.15 * np.cos(1.8 * x - roughness * y)
    valley = -0.35 * np.exp(-12 * (y - 0.35 * np.sin(2 * x)) ** 2)
    noise = rng.normal(0, 0.05, (size, size))
    return base + waves + valley + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mock_tiles.npz")
    parser.add_argument("--n", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-size", type=int, default=16)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    hr = args.lr_size * args.scale
    x_lr, a_hr, y_hr, region, tile_id = [], [], [], [], []
    for i in range(args.n):
        reg = REGIONS[i % len(REGIONS)]
        rough = {"pine_island": 4.5, "thwaites": 5.2, "rutford": 3.2, "wilkes": 6.1, "ross": 2.8}[reg]
        y = terrain(hr, rng, rough).astype("float32")
        y = (y - y.mean()) / (y.std() + 1e-6)
        low = resize(y, (args.lr_size, args.lr_size), anti_aliasing=True).astype("float32")
        yy, xx = np.gradient(y)
        surface = y + 0.35 * np.exp(-np.abs(y)) + rng.normal(0, 0.03, y.shape)
        velocity = np.sqrt(xx**2 + yy**2) + rng.normal(0, 0.01, y.shape)
        accumulation = np.sin(np.linspace(0, np.pi, hr))[None, :] + rng.normal(0, 0.02, y.shape)
        aux = np.stack([surface, velocity, accumulation], axis=0).astype("float32")
        aux = (aux - aux.mean(axis=(1, 2), keepdims=True)) / (aux.std(axis=(1, 2), keepdims=True) + 1e-6)
        x_lr.append(low[None])
        a_hr.append(aux)
        y_hr.append(y[None])
        region.append(reg)
        tile_id.append(f"{reg}_{i:05d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x_lr=np.stack(x_lr), a_hr=np.stack(a_hr), y_hr=np.stack(y_hr), region=np.asarray(region), tile_id=np.asarray(tile_id))
    print(f"wrote {args.out} with {args.n} tiles")


if __name__ == "__main__":
    main()
