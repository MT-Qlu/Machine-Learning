from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf

from .config import CONFIG, DiffusionConfig
from .engine import DiffusionEngine
from .model import build_unet
from .utils import save_image_grid


def load_model(config: DiffusionConfig = CONFIG, checkpoint_path: Optional[Path] = None) -> tf.keras.Model:
    model = build_unet(image_size=config.image_size, channels=config.channels)
    path = checkpoint_path or (config.checkpoint_dir / "ddpm.weights.h5")
    model.load_weights(str(path))
    return model


def generate_samples(
    config: DiffusionConfig = CONFIG,
    num_images: int = 16,
    checkpoint_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> tf.Tensor:
    model = load_model(config, checkpoint_path)
    engine = DiffusionEngine(config)
    samples = engine.sample(model, num_images)

    if output_path is not None:
        save_image_grid(samples.numpy(), output_path, nrow=int(num_images ** 0.5))

    return samples
