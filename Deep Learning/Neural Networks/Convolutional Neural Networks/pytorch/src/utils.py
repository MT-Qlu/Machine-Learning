"""Helper utilities for PyTorch CNN training."""
from __future__ import annotations

import math
from typing import Dict

import torch


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Return accuracy for a batch."""
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def summarize_metrics(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> str:
    return (
        f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
    )


def format_time(seconds: float) -> str:
    minutes = math.floor(seconds / 60)
    secs = seconds - minutes * 60
    return f"{minutes:d}m {secs:.1f}s"
