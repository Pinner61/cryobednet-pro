from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from cryobednet.visualization import save_error_hist, save_prediction_panel, save_transect, save_history_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="holdout")
    args = parser.parse_args()
    run = Path(args.run)
    z = np.load(run / f"predictions_{args.split}.npz")
    pred, target, bicubic = z["pred"], z["target"], z["bicubic"]
    if (run / "history.csv").exists():
        save_history_plot(run / "history.csv", run / "loss_curve.png")
    save_prediction_panel(pred, target, bicubic, run / f"prediction_panel_{args.split}.png")
    save_transect(pred, target, bicubic, run / f"transect_{args.split}.png")
    save_error_hist(pred, target, run / f"error_hist_{args.split}.png")
    print(f"figures written to {run}")


if __name__ == "__main__":
    main()
