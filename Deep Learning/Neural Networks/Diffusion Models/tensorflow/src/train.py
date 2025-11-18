from __future__ import annotations

from pathlib import Path
from typing import Dict

import tensorflow as tf
from tqdm.auto import tqdm

from .config import CONFIG, DiffusionConfig
from .data import load_datasets
from .engine import DiffusionEngine
from .model import build_unet
from .utils import save_image_grid, seed_everything, write_metrics


def _ensure_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train(config: DiffusionConfig = CONFIG) -> Dict[str, list[float]]:
    """Train the TensorFlow DDPM on Fashion-MNIST."""

    seed_everything(config.seed)
    _ensure_artifacts(config.artifact_dir)
    _ensure_artifacts(config.checkpoint_dir)

    strategy = config.strategy()
    engine = DiffusionEngine(config)
    train_ds, val_ds = load_datasets(config)

    with strategy.scope():
        model = build_unet(image_size=config.image_size, channels=config.channels)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    metrics: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        train_loss = tf.metrics.Mean()
        for batch in tqdm(train_ds, desc=f"Epoch {epoch+1}/{config.epochs}"):
            t = engine.sample_timesteps(tf.shape(batch)[0])
            with tf.GradientTape() as tape:
                loss = engine.loss(model, batch, t)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(loss)

        metrics["train_loss"].append(float(train_loss.result().numpy()))

        val_loss_metric = tf.metrics.Mean()
        for batch in val_ds:
            t = engine.sample_timesteps(tf.shape(batch)[0])
            x_t, noise = engine.q_sample(batch, t)
            pred = model([x_t, t], training=False)
            loss = tf.reduce_mean(tf.square(pred - noise))
            val_loss_metric.update_state(loss)

        metrics["val_loss"].append(float(val_loss_metric.result().numpy()))

    model.save_weights(str(config.checkpoint_dir / "ddpm.weights.h5"))

    samples = engine.sample(model, config.sample_batch_size)
    save_image_grid(samples.numpy(), config.artifact_dir / "ddpm_samples.png", nrow=8)
    write_metrics(metrics, config.metrics_file)

    return metrics
