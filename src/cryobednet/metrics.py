import math
import numpy as np
from skimage.metrics import structural_similarity


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(np.max(target) - np.min(target) + 1e-8)
    return float(20 * math.log10(data_range / math.sqrt(mse)))


def ssim(pred, target):
    vals = []
    for p, t in zip(pred[:, 0], target[:, 0]):
        data_range = float(t.max() - t.min() + 1e-8)
        vals.append(structural_similarity(t, p, data_range=data_range))
    return float(np.mean(vals))


def slope_rmse(pred, target):
    px = np.diff(pred, axis=-1)
    tx = np.diff(target, axis=-1)
    py = np.diff(pred, axis=-2)
    ty = np.diff(target, axis=-2)
    return float(np.sqrt(np.mean((px - tx) ** 2) + np.mean((py - ty) ** 2)))


def compute_metrics(pred, target):
    return {"mae": mae(pred, target), "rmse": rmse(pred, target), "psnr": psnr(pred, target), "ssim": ssim(pred, target), "slope_rmse": slope_rmse(pred, target)}
