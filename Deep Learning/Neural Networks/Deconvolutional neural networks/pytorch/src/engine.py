"""Training and evaluation loops for the autoencoder."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import compute_batch_psnr


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    total = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        targets = inputs

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_psnr += compute_batch_psnr(outputs.detach(), targets) * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    epoch_psnr = running_psnr / total
    return {"loss": epoch_loss, "psnr": epoch_psnr}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    total = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            targets = inputs

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_psnr += compute_batch_psnr(outputs, targets) * batch_size
            total += batch_size

    epoch_loss = running_loss / total
    epoch_psnr = running_psnr / total
    return {"loss": epoch_loss, "psnr": epoch_psnr}
