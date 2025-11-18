from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import CONFIG, GANConfig
from .engine import GANEngine
from .model import Generator
from .utils import save_image_grid


def load_generator(config: GANConfig = CONFIG, checkpoint_path: Optional[Path] = None) -> Generator:
    generator = Generator(latent_dim=config.latent_dim, channels=config.channels)
    path = checkpoint_path or config.generator_ckpt
    state = torch.load(path, map_location=config.device)
    generator.load_state_dict(state)
    generator.to(config.device)
    generator.eval()
    return generator


@torch.no_grad()
def generate_samples(
    config: GANConfig = CONFIG,
    num_images: int = 16,
    checkpoint_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> torch.Tensor:
    generator = load_generator(config, checkpoint_path)
    engine = GANEngine(config)
    noise = engine.sample_noise(num_images)
    samples = generator(noise).cpu()

    if output_path is not None:
        save_image_grid(samples, output_path, nrow=int(num_images ** 0.5))

    return samples
