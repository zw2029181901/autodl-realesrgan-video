"""RRDB architecture compatible with the official RealESRGAN_x4plus checkpoint.

Architecture reference: XPixelGroup/BasicSR basicsr/archs/rrdbnet_arch.py (Apache-2.0).
This implementation preserves its checkpoint key layout; see THIRD_PARTY.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, features=64, growth=32):
        super().__init__()
        for index in range(1, 6):
            setattr(self, f"conv{index}", nn.Conv2d(features+(index-1)*growth, growth if index < 5 else features, 3, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        values = [x]
        for index in range(1, 5):
            values.append(self.lrelu(getattr(self, f"conv{index}")(torch.cat(values, 1))))
        return self.conv5(torch.cat(values, 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(23)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        features = self.conv_first(x)
        features = features + self.conv_body(self.body(features))
        features = self.lrelu(self.conv_up1(F.interpolate(features, scale_factor=2, mode="nearest")))
        features = self.lrelu(self.conv_up2(F.interpolate(features, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(features)))
