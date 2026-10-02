import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(256, condition_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        condition_map = self.encoder(x)

        pooled = self.pool(condition_map)
        pooled = pooled.flatten(1)

        condition_vector = self.projection(pooled)

        return condition_map, condition_vector