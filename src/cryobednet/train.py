from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config
from .data.dataset import CryoTileDataset, TileBundle
from .data.splits import region_holdout_split
from .losses import gan_d_loss, gan_g_loss, generator_loss
from .models.discriminator import PatchDiscriminator
from .models.generator import build_generator
from .utils import device_from_config, save_json, seed_everything


def maybe_mlflow(cfg):
    if not cfg.get("mlflow", {}).get("enabled", False):
        return None
    try:
        import mlflow
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "CryoBedNet"))
        return mlflow
    except Exception:
        return None


def train_one_epoch(model, disc, loader, opt_g, opt_d, scaler, cfg, device):
    model.train()
    if disc is not None:
        disc.train()
    total = 0.0
    amp = cfg["train"].get("amp", False) and device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        x_lr = batch["x_lr"].to(device)
        a_hr = batch["a_hr"].to(device)
        y_hr = batch["y_hr"].to(device)
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(x_lr, a_hr)
            loss = generator_loss(pred, y_hr, cfg["train"].get("grad_loss_weight", 0.15))
            if disc is not None:
                loss = loss + cfg["train"].get("adv_weight", 0.0005) * gan_g_loss(disc(pred))
        scaler.scale(loss).backward()
        scaler.step(opt_g)
        scaler.update()
        if disc is not None:
            opt_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                fake = model(x_lr, a_hr).detach()
                d_loss = gan_d_loss(disc(y_hr), disc(fake))
            d_loss.backward()
            opt_d.step()
        total += float(loss.detach().cpu()) * x_lr.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="validate", leave=False):
        x_lr = batch["x_lr"].to(device)
        a_hr = batch["a_hr"].to(device)
        y_hr = batch["y_hr"].to(device)
        pred = model(x_lr, a_hr)
        loss = generator_loss(pred, y_hr, cfg["train"].get("grad_loss_weight", 0.15))
        total += float(loss.detach().cpu()) * x_lr.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    out = ensure_dir(cfg["output_dir"])
    device = device_from_config(cfg["train"].get("device", "auto"))
    bundle = TileBundle.load(cfg["data"]["path"])
    train_idx, val_idx, holdout_idx = region_holdout_split(bundle.region, cfg["split"]["holdout_region"], cfg["split"]["val_fraction"], cfg.get("seed", 42))
    save_json(out / "splits.json", {"train": len(train_idx), "val": len(val_idx), "holdout": len(holdout_idx)})
    train_loader = DataLoader(CryoTileDataset(bundle, train_idx), batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"].get("num_workers", 0), pin_memory=device.type == "cuda")
    val_loader = DataLoader(CryoTileDataset(bundle, val_idx), batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"].get("num_workers", 0), pin_memory=device.type == "cuda")
    model = build_generator(cfg["model"]).to(device)
    disc = PatchDiscriminator().to(device) if cfg["train"].get("adversarial", False) else None
    opt_g = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg["train"]["lr"]) if disc is not None else None
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"].get("amp", False) and device.type == "cuda")
    mlflow = maybe_mlflow(cfg)
    if mlflow:
        mlflow.start_run(run_name=cfg.get("run_name", Path(cfg["output_dir"]).name))
        mlflow.log_params({"model": cfg["model"]["name"], "holdout_region": cfg["split"]["holdout_region"]})
    history, best = [], float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, disc, train_loader, opt_g, opt_d, scaler, cfg, device)
        val_loss = validate(model, val_loader, cfg, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        save_json(out / "metrics_val.json", {"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch})
        if mlflow:
            mlflow.log_metrics({"val_loss": val_loss, "train_loss": train_loss}, step=epoch)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "checkpoint_best.pt")
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    if mlflow:
        mlflow.log_artifact(str(out / "history.csv"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
