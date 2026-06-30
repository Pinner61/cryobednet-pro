from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=48):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_channels
        for _ in range(3):
            layers += [nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, inplace=True)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
