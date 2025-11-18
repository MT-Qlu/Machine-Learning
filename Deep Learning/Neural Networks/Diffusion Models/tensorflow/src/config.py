from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


@dataclass
class DiffusionConfig:
    """Hyperparameters and paths for the TensorFlow DDPM."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    artifact_dir: Path = project_root / "artifacts" / "tensorflow_diffusion"
    metrics_file: Path = artifact_dir / "metrics.json"
    checkpoint_dir: Path = artifact_dir / "checkpoints"
    batch_size: int = 128
    image_size: int = 28
    channels: int = 1
    learning_rate: float = 2e-4
    epochs: int = 20
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sample_steps: int = 100
    sample_batch_size: int = 64
    seed: int = 42

    def strategy(self) -> tf.distribute.Strategy:
        return tf.distribute.get_strategy()


CONFIG = DiffusionConfig()
