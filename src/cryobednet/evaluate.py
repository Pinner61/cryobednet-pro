from __future__ import annotations
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config
from .data.dataset import CryoTileDataset, TileBundle
from .data.splits import region_holdout_split
from .metrics import compute_metrics
from .models.generator import build_generator
from .utils import device_from_config, save_json
from .visualization import save_error_hist, save_history_plot, save_prediction_panel, save_transect


def indices_for_split(bundle, cfg, split):
    train_idx, val_idx, holdout_idx = region_holdout_split(bundle.region, cfg["split"]["holdout_region"], cfg["split"]["val_fraction"], cfg.get("seed", 42))
    return {"train": train_idx, "val": val_idx, "holdout": holdout_idx}[split]


@torch.no_grad()
def predict(model, loader, device):
    preds, targets, bicubics, ids = [], [], [], []
    model.eval()
    for batch in tqdm(loader, desc="predict", leave=False):
        x_lr = batch["x_lr"].to(device)
        a_hr = batch["a_hr"].to(device)
        y_hr = batch["y_hr"].to(device)
        pred = model(x_lr, a_hr)
        bicubic = F.interpolate(x_lr, size=y_hr.shape[-2:], mode="bicubic", align_corners=False)
        preds.append(pred.cpu().numpy())
        targets.append(y_hr.cpu().numpy())
        bicubics.append(bicubic.cpu().numpy())
        ids.extend(batch["tile_id"])
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(bicubics), np.asarray(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="holdout", choices=["train", "val", "holdout"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["output_dir"])
    device = device_from_config(cfg["train"].get("device", "auto"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    bundle = TileBundle.load(cfg["data"]["path"])
    idx = indices_for_split(bundle, cfg, args.split)
    loader = DataLoader(CryoTileDataset(bundle, idx), batch_size=cfg["data"]["batch_size"], shuffle=False)
    model = build_generator(cfg["model"]).to(device)
    ckpt = torch.load(out / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    pred, target, bicubic, tile_id = predict(model, loader, device)
    metrics = {"model": compute_metrics(pred, target), "bicubic": compute_metrics(bicubic, target), "split": args.split, "n_tiles": int(len(idx))}
    save_json(out / f"metrics_{args.split}.json", metrics)
    np.savez_compressed(out / f"predictions_{args.split}.npz", pred=pred, target=target, bicubic=bicubic, tile_id=tile_id)
    if (out / "history.csv").exists():
        save_history_plot(out / "history.csv", out / "loss_curve.png")
    save_prediction_panel(pred, target, bicubic, out / f"prediction_panel_{args.split}.png")
    save_transect(pred, target, bicubic, out / f"transect_{args.split}.png")
    save_error_hist(pred, target, out / f"error_hist_{args.split}.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
