from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import CONFIG, DiffusionConfig
from .engine import DiffusionEngine
from .model import UNet
from .utils import save_image_grid


def load_model(config: DiffusionConfig = CONFIG, checkpoint_path: Optional[Path] = None) -> UNet:
    model = UNet(channels=config.channels)
    path = checkpoint_path or config.checkpoint_file
    state = torch.load(path, map_location=config.device)
    model.load_state_dict(state)
    model.to(config.device)
    model.eval()
    return model


@torch.no_grad()
def generate_samples(
    config: DiffusionConfig = CONFIG,
    num_images: int = 16,
    checkpoint_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> torch.Tensor:
    model = load_model(config, checkpoint_path)
    engine = DiffusionEngine(config, config.device)
    images = engine.sample(model, num_images, config.image_size)

    if output_path is not None:
        save_image_grid(images, output_path, nrow=int(num_images ** 0.5))

    return images
