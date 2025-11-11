"""Data loading utilities for the PyTorch CNN baseline."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import CNNConfig, CONFIG


def _build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_datasets(config: CNNConfig = CONFIG) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    transform = _build_transforms()
    train_ds = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return train_ds, test_ds


def create_dataloaders(
    config: CNNConfig = CONFIG,
) -> Tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_ds, test_ds = load_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.type != "cpu",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.type != "cpu",
    )
    return train_loader, test_loader
