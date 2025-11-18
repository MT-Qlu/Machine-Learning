import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Positional encoding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device).float()
        emb = torch.exp(-math.log(10000) * emb / (half_dim - 1))
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """A lightweight UNet backbone for 28x28 Fashion-MNIST."""

    def __init__(self, channels: int = 1, time_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.initial = nn.Conv2d(channels, base_channels, kernel_size=3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.downsample1 = Downsample(base_channels * 2)
        self.downsample2 = Downsample(base_channels * 4)

        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_dim)
        self.upsample1 = Upsample(base_channels * 4)
        self.upsample2 = Upsample(base_channels * 2)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_embedding(timesteps)

        x1 = self.initial(x)
        x2 = self.down1(x1, t)
        x3 = self.downsample1(x2)
        x4 = self.down2(x3, t)
        x5 = self.downsample2(x4)

        mid = self.mid1(x5, t)
        mid = self.mid2(mid, t)

        u1 = self.upsample1(mid)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.up1(u1, t)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2, t)

        out = self.final(u2 + x1)
        return out
