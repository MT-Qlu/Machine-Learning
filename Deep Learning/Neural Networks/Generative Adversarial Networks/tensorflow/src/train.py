from __future__ import annotations

from pathlib import Path
from typing import Dict

import tensorflow as tf
from tqdm.auto import tqdm

from .config import CONFIG, GANConfig
from .data import load_dataset
from .engine import GANEngine
from .model import build_discriminator, build_generator
from .utils import save_image_grid, seed_everything, write_metrics


def _ensure_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train(config: GANConfig = CONFIG) -> Dict[str, list[float]]:
    seed_everything(config.seed)
    _ensure_artifacts(config.artifact_dir)

    dataset, dataset_size = load_dataset(config)

    generator = build_generator(latent_dim=config.latent_dim, channels=config.channels)
    discriminator = build_discriminator(channels=config.channels)

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
    )

    engine = GANEngine(config)
    metrics: Dict[str, list[float]] = {"g_loss": [], "d_loss": []}

    for epoch in range(config.epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        progress = tqdm(dataset, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for real_images in progress:
            d_loss = engine.discriminator_step(discriminator, generator, discriminator_optimizer, real_images)
            g_loss = engine.generator_step(discriminator, generator, generator_optimizer, real_images.shape[0])
            d_epoch_loss += d_loss * real_images.shape[0]
            g_epoch_loss += g_loss * real_images.shape[0]
            progress.set_postfix(d_loss=d_loss, g_loss=g_loss)

        metrics["d_loss"].append(d_epoch_loss / dataset_size)
        metrics["g_loss"].append(g_epoch_loss / dataset_size)

    generator.save_weights(str(config.generator_ckpt))
    discriminator.save_weights(str(config.discriminator_ckpt))

    samples = generator(engine.sample_noise(config.sample_batch_size), training=False)
    save_image_grid(samples.numpy(), config.grid_path, nrow=8)

    write_metrics(metrics, config.metrics_file)
    return metrics
