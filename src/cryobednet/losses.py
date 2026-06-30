import torch
import torch.nn.functional as F


def terrain_gradient(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def gradient_loss(pred, target):
    px, py = terrain_gradient(pred)
    tx, ty = terrain_gradient(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def generator_loss(pred, target, grad_weight=0.15):
    return F.l1_loss(pred, target) + grad_weight * gradient_loss(pred, target)


def gan_g_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def gan_d_loss(real_logits, fake_logits):
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real + fake)
