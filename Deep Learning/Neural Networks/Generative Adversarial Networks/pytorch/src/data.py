from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import GANConfig
from .utils import seed_everything


def load_dataloaders(config: GANConfig) -> Tuple[DataLoader, DataLoader]:
def load_dataloaders(config: GANConfig) -> Tuple[DataLoader, int]:
    seed_everything(config.seed)
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_ds = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, len(train_ds)
