"""TensorFlow data pipeline for the deconvolutional autoencoder."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .config import CONFIG, DeconvConfig


def load_datasets(config: DeconvConfig = CONFIG) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    (train_x, _), (test_x, _) = tf.keras.datasets.fashion_mnist.load_data(
        path=str(config.data_dir / "fashion_mnist.npz")
    )

    train_ds = _build_dataset(train_x, config, training=True)
    test_ds = _build_dataset(test_x, config, training=False)
    return train_ds, test_ds


def _build_dataset(images: tf.Tensor, config: DeconvConfig, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(images[..., tf.newaxis])
    dataset = dataset.map(_normalise_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=10_000, seed=config.seed)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _normalise_pair(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image, image
