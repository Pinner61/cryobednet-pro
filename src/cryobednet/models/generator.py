import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNetSR(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, upscale=4):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x_lr, a_hr):
        x = F.interpolate(x_lr, size=a_hr.shape[-2:], mode="bicubic", align_corners=False)
        z = torch.cat([x, a_hr], dim=1)
        e1 = self.enc1(z)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return x + self.out(d1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth=24):
        super().__init__()
        self.c1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(channels + growth * 2, growth, 3, padding=1)
        self.c4 = nn.Conv2d(channels + growth * 3, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.c4(torch.cat([x, x1, x2, x3], 1))
        return x + 0.2 * x4


class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b1 = ResidualDenseBlock(channels)
        self.b2 = ResidualDenseBlock(channels)
        self.b3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + 0.2 * self.b3(self.b2(self.b1(x)))


class RRDBGenerator(nn.Module):
    def __init__(self, in_channels=4, base_channels=48, num_blocks=6, upscale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(base_channels) for _ in range(num_blocks)])
        self.trunk = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x_lr, a_hr):
        x = F.interpolate(x_lr, size=a_hr.shape[-2:], mode="bicubic", align_corners=False)
        z = torch.cat([x, a_hr], dim=1)
        feat = self.head(z)
        body = self.trunk(self.body(feat))
        return x + self.refine(feat + body)


def build_generator(cfg: dict) -> nn.Module:
    name = cfg.get("name", "residual_unet")
    if name == "residual_unet":
        return ResidualUNetSR(cfg.get("in_channels", 4), cfg.get("base_channels", 32), cfg.get("upscale", 4))
    if name == "rrdb":
        return RRDBGenerator(cfg.get("in_channels", 4), cfg.get("base_channels", 48), cfg.get("num_blocks", 6), cfg.get("upscale", 4))
    raise ValueError(f"Unknown model: {name}")
