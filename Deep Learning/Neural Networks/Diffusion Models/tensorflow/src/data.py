from typing import Tuple

import tensorflow as tf

from .config import DiffusionConfig
from .utils import seed_everything


def _prepare_dataset(images: tf.Tensor, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(images)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=images.shape[0])
    dataset = dataset.map(lambda x: (tf.expand_dims(x, axis=-1) / 127.5) - 1.0)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_datasets(config: DiffusionConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    seed_everything(config.seed)
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data(path=str(config.data_dir / "fashion-mnist.npz"))
    return _prepare_dataset(x_train, config.batch_size, True), _prepare_dataset(x_test, config.batch_size, False)
