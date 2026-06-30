from __future__ import annotations
import argparse
import json
from pathlib import Path


def row(name, m):
    return f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['slope_rmse']:.4f} |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/mock_cpu/metrics_holdout.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    block = "\n## Initial holdout metrics\n\n| Method | MAE | RMSE | PSNR | SSIM | Slope RMSE |\n|---|---:|---:|---:|---:|---:|\n"
    block += row("CryoBedNet", metrics["model"]) + "\n"
    block += row("Bicubic", metrics["bicubic"]) + "\n"
    p = Path(args.readme)
    text = p.read_text()
    if "## Initial holdout metrics" not in text:
        p.write_text(text + block)
    print("README updated")


if __name__ == "__main__":
    main()
