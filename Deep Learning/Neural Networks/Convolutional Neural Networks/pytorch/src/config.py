"""Configuration for the PyTorch CNN baseline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def _detect_device() -> torch.device:
    """Return the best available torch device with MPS support prioritised."""
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(slots=True, frozen=True)
class CNNConfig:
    """Hyperparameters and file-system layout for training."""

    data_dir: Path = Path("artifacts/pytorch_cnn/data")
    artifact_dir: Path = Path("artifacts/pytorch_cnn")
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    device: torch.device = _detect_device()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = CNNConfig()
CONFIG.ensure_dirs()
