from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DiffusionConfig:
    """Centralised hyperparameters and paths for the PyTorch DDPM."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    artifact_dir: Path = project_root / "artifacts" / "pytorch_diffusion"
    metrics_file: Path = artifact_dir / "metrics.json"
    checkpoint_file: Path = artifact_dir / "ddpm_fashion_mnist.pt"
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 28
    channels: int = 1
    epochs: int = 20
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (1e-4, 0.02)
    timesteps: int = 1000
    sample_steps: int = 100
    sample_batch_size: int = 64
    seed: int = 42

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


CONFIG = DiffusionConfig()
