from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_history_plot(history_csv: str | Path, out: str | Path):
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_prediction_panel(pred, target, bicubic, out: str | Path, idx=0):
    mats = [target[idx, 0], pred[idx, 0], bicubic[idx, 0], pred[idx, 0] - target[idx, 0]]
    titles = ["target", "prediction", "bicubic", "error"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_transect(pred, target, bicubic, out: str | Path, idx=0):
    row = pred.shape[-2] // 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(target[idx, 0, row], label="target")
    ax.plot(pred[idx, 0, row], label="prediction")
    ax.plot(bicubic[idx, 0, row], label="bicubic")
    ax.set_xlabel("pixel")
    ax.set_ylabel("normalized elevation")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_error_hist(pred, target, out: str | Path):
    err = (pred - target).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=80)
    ax.set_xlabel("prediction error")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
