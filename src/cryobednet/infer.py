from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .data.dataset import CryoTileDataset, TileBundle
from .models.generator import build_generator
from .utils import device_from_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = device_from_config(cfg["train"].get("device", "auto"))
    bundle = TileBundle.load(args.data)
    loader = DataLoader(CryoTileDataset(bundle, np.arange(len(bundle.region))), batch_size=cfg["data"].get("batch_size", 8))
    model = build_generator(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    preds = []
    for batch in loader:
        preds.append(model(batch["x_lr"].to(device), batch["a_hr"].to(device)).cpu().numpy())
    np.savez_compressed(args.out, pred=np.concatenate(preds), tile_id=bundle.tile_id)


if __name__ == "__main__":
    main()
