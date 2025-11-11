"""Training and evaluation loops."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import calculate_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += calculate_accuracy(outputs, targets) * inputs.size(0)
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            running_correct += calculate_accuracy(outputs, targets) * inputs.size(0)
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}
