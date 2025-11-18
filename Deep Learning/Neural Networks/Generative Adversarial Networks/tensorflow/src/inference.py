from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf

from .config import CONFIG, GANConfig
from .engine import GANEngine
from .model import build_generator
from .utils import save_image_grid


def load_generator(config: GANConfig = CONFIG, checkpoint_path: Optional[Path] = None) -> tf.keras.Model:
    generator = build_generator(latent_dim=config.latent_dim, channels=config.channels)
    generator(tf.zeros((1, config.latent_dim)))
    path = checkpoint_path or config.generator_ckpt
    generator.load_weights(str(path))
    return generator


def generate_samples(
    config: GANConfig = CONFIG,
    num_images: int = 16,
    checkpoint_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> tf.Tensor:
    generator = load_generator(config, checkpoint_path)
    engine = GANEngine(config)
    noise = engine.sample_noise(num_images)
    samples = generator(noise, training=False)

    if output_path is not None:
        save_image_grid(samples.numpy(), output_path, nrow=int(num_images ** 0.5))

    return samples
