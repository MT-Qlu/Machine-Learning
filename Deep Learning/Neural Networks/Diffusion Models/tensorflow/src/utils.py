import json
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> tf.Tensor:
    return tf.linspace(beta_start, beta_end, timesteps)


def extract(a: tf.Tensor, t: tf.Tensor, shape: tf.TensorShape) -> tf.Tensor:
    batch_size = tf.shape(t)[0]
    gather = tf.gather(a, indices=t)
    reshape = tf.reshape(gather, (batch_size,) + (1,) * (len(shape) - 1))
    return reshape


def write_metrics(metrics: Dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def save_image_grid(images: np.ndarray, path: Path, nrow: int = 8) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    num_images = images.shape[0]
    ncol = nrow
    nrow = max(1, int(np.ceil(num_images / ncol)))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = np.array(axes).reshape(nrow, ncol)

    for idx in range(nrow * ncol):
        ax = axes[idx // ncol, idx % ncol]
        ax.axis("off")
        if idx < num_images:
            img = images[idx]
            img = (img + 1.0) / 2.0
            ax.imshow(img.squeeze(), cmap="gray")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
