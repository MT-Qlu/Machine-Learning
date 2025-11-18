from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import DiffusionConfig
from .utils import extract, linear_beta_schedule


@dataclass
class DiffusionState:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_hats: torch.Tensor


class DiffusionEngine:
    """Utility class encapsulating forward and reverse diffusion operations."""

    def __init__(self, config: DiffusionConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or config.device
        betas = linear_beta_schedule(config.timesteps, *config.betas)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)

        self.state = DiffusionState(
            betas=betas.to(self.device),
            alphas=alphas.to(self.device),
            alpha_hats=alpha_hats.to(self.device),
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.config.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = torch.sqrt(extract(self.state.alpha_hats, t, x_start.shape))
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - extract(self.state.alpha_hats, t, x_start.shape))
        x_t = sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def p_losses(self, model: torch.nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_t, noise = self.q_sample(x_start, t)
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, batch_size: int, img_size: int) -> torch.Tensor:
        model.eval()
        img = torch.randn(batch_size, self.config.channels, img_size, img_size, device=self.device)

        betas = self.state.betas
        alphas = self.state.alphas
        alpha_hats = self.state.alpha_hats

        step_indices = torch.linspace(
            0,
            self.config.timesteps - 1,
            self.config.sample_steps,
            dtype=torch.long,
            device=self.device,
        )

        for idx in reversed(step_indices):
            t = int(idx.item())
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_hat_t = alpha_hats[t]

            pred_noise = model(img, timesteps)

            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (
                1 / torch.sqrt(alpha_t)
                * (img - (beta_t / torch.sqrt(1 - alpha_hat_t)) * pred_noise)
                + torch.sqrt(beta_t) * noise
            )

        return img
