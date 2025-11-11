"""Configuration for the PyTorch deconvolutional autoencoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def _detect_device() -> torch.device:
    """Detect the best available device with MPS priority."""
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(slots=True, frozen=True)
class DeconvConfig:
    data_dir: Path = Path("artifacts/pytorch_deconv/data")
    artifact_dir: Path = Path("artifacts/pytorch_deconv")
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 2
    seed: int = 42
    device: torch.device = _detect_device()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = DeconvConfig()
CONFIG.ensure_dirs()
