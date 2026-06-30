import argparse
import json
import numpy as np
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-std", type=float, default=0.04)
    p.add_argument("--min-relief", type=float, default=0.15)
    args = p.parse_args()

    z = np.load(args.infile, allow_pickle=True)
    y = z["y_hr"]
    region = z["region"].astype(str)

    flat = y.reshape(y.shape[0], -1)
    std = flat.std(axis=1)
    relief = flat.max(axis=1) - flat.min(axis=1)

    keep = (std >= args.min_std) & (relief >= args.min_relief)

    payload = {}
    for k in z.files:
        arr = z[k]
        if len(arr) == len(keep):
            payload[k] = arr[keep]
        else:
            payload[k] = arr

    np.savez_compressed(args.out, **payload)

    counts = Counter(payload["region"].astype(str))
    summary = {
        "input_tiles": int(len(keep)),
        "kept_tiles": int(keep.sum()),
        "removed_tiles": int((~keep).sum()),
        "min_std": args.min_std,
        "min_relief": args.min_relief,
        "recommended_holdout": max(counts, key=counts.get),
        "region_counts": dict(sorted(counts.items())),
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
