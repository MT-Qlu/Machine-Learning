from typing import Tuple

import tensorflow as tf

from .config import GANConfig
from .utils import seed_everything


def load_dataset(config: GANConfig) -> Tuple[tf.data.Dataset, int]:
    seed_everything(config.seed)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    data_path = config.data_dir / "fashion_mnist.npz"
    (train_images, _), _ = tf.keras.datasets.fashion_mnist.load_data(path=str(data_path))

    train_images = train_images.astype("float32")
    train_images = (train_images / 127.5) - 1.0
    train_images = train_images[..., None]

    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(buffer_size=train_images.shape[0], seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, int(train_images.shape[0])
