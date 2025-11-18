from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DiffusionConfig
from .utils import seed_everything


def _fashion_mnist_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # to [-1, 1]
        ]
    )


def load_dataloaders(config: DiffusionConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for Fashion-MNIST."""

    seed_everything(config.seed)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    transform = _fashion_mnist_transform(config.image_size)

    train_ds = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    val_ds = datasets.FashionMNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
