"""Data pipeline for TensorFlow CNN baseline."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .config import CNNConfig, CONFIG


def load_datasets(config: CNNConfig = CONFIG) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data(path=str(config.data_dir / "fashion_mnist.npz"))

    train_ds = _build_dataset(train_x, train_y, config, training=True)
    test_ds = _build_dataset(test_x, test_y, config, training=False)
    return train_ds, test_ds


def _build_dataset(
    images: tf.Tensor,
    labels: tf.Tensor,
    config: CNNConfig,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images[..., tf.newaxis], labels))
    dataset = dataset.map(_normalise, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=10_000, seed=config.seed)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _normalise(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image, label
