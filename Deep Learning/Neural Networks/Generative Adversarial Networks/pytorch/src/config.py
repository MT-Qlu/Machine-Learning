from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GANConfig:
    """Configuration for the DCGAN baseline."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    artifact_dir: Path = project_root / "artifacts" / "pytorch_gan"
    generator_ckpt: Path = artifact_dir / "generator.pt"
    discriminator_ckpt: Path = artifact_dir / "discriminator.pt"
    metrics_file: Path = artifact_dir / "metrics.json"
    grid_path: Path = artifact_dir / "gan_samples.png"

    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 28
    channels: int = 1
    latent_dim: int = 100
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    epochs: int = 50
    sample_batch_size: int = 64
    seed: int = 42

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


CONFIG = GANConfig()
