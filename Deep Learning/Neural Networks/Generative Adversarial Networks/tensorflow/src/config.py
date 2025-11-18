from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


@dataclass
class GANConfig:
    """Configuration for the TensorFlow DCGAN baseline."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    artifact_dir: Path = project_root / "artifacts" / "tensorflow_gan"
    generator_ckpt: Path = artifact_dir / "generator"
    discriminator_ckpt: Path = artifact_dir / "discriminator"
    metrics_file: Path = artifact_dir / "metrics.json"
    grid_path: Path = artifact_dir / "gan_samples.png"

    batch_size: int = 128
    image_size: int = 28
    channels: int = 1
    latent_dim: int = 100
    learning_rate: float = 2e-4
    beta_1: float = 0.5
    beta_2: float = 0.999
    epochs: int = 50
    sample_batch_size: int = 64
    seed: int = 42

    def strategy(self) -> tf.distribute.Strategy:
        return tf.distribute.get_strategy()


CONFIG = GANConfig()
