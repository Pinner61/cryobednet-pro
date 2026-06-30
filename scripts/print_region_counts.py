from __future__ import annotations

import argparse
from collections import Counter
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    args = parser.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    for name, count in Counter(z["region"].tolist()).most_common():
        print(f"{name}\t{count}")


if __name__ == "__main__":
    main()
