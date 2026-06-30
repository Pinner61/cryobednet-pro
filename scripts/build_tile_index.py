from __future__ import annotations
import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", nargs="+", required=True)
    parser.add_argument("--out", default="data/tile_index.json")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()
    import rasterio
    records = []
    with rasterio.open(args.rasters[0]) as src:
        width, height = src.width, src.height
        crs = str(src.crs)
    for y in range(0, height - args.tile_size + 1, args.stride):
        for x in range(0, width - args.tile_size + 1, args.stride):
            records.append({"x": x, "y": y, "w": args.tile_size, "h": args.tile_size, "crs": crs})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} tiles to {args.out}")


if __name__ == "__main__":
    main()
