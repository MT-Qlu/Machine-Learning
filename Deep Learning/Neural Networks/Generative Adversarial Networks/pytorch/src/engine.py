from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .config import GANConfig


class GANEngine:
    def __init__(self, config: GANConfig) -> None:
        self.config = config
        self.device = config.device

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.config.latent_dim, 1, 1, device=self.device)

    def discriminator_step(
        self,
        discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        real_images: torch.Tensor,
    ) -> float:
        batch_size = real_images.size(0)
        noise = self.sample_noise(batch_size)
        fake_images = generator(noise).detach()

        optimizer.zero_grad(set_to_none=True)
        real_preds = discriminator(real_images)
        fake_preds = discriminator(fake_images)

        real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def generator_step(
        self,
        discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ) -> float:
        noise = self.sample_noise(batch_size)
        optimizer.zero_grad(set_to_none=True)
        fake_images = generator(noise)
        preds = discriminator(fake_images)
        loss = F.binary_cross_entropy_with_logits(preds, torch.ones_like(preds))
        loss.backward()
        optimizer.step()
        return float(loss.item())
