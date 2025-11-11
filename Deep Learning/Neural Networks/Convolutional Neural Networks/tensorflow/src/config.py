"""Configuration for the TensorFlow CNN baseline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tensorflow as tf


@dataclass(slots=True, frozen=True)
class CNNConfig:
    """Hyperparameters and file layout for TensorFlow training."""

    data_dir: Path = Path("artifacts/tensorflow_cnn/data")
    artifact_dir: Path = Path("artifacts/tensorflow_cnn")
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device_preference: Literal["MPS", "GPU", "CPU"] = "MPS"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


def configure_devices(config: CNNConfig) -> None:
    """Pin TensorFlow to the requested device preference."""
    physical = tf.config.list_physical_devices()
    if config.device_preference == "MPS":
        mps_devices = [d for d in physical if d.device_type.upper() == "GPU" and "MPS" in d.name.upper()]
        if mps_devices:
            tf.config.set_visible_devices(mps_devices, "GPU")
            return
    if config.device_preference in {"MPS", "GPU"}:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus, "GPU")
            return
    tf.config.set_visible_devices([], "GPU")


CONFIG = CNNConfig()
CONFIG.ensure_dirs()
configure_devices(CONFIG)
