import numpy as np
from cryobednet.metrics import compute_metrics


def test_compute_metrics():
    target = np.zeros((2, 1, 16, 16), dtype="float32")
    pred = np.ones_like(target) * 0.1
    m = compute_metrics(pred, target)
    assert m["mae"] > 0
    assert m["rmse"] > 0
    assert "ssim" in m
