"""Utility helpers for deconvolutional training."""
from __future__ import annotations

import math
from typing import Dict

import torch


_EPS = 1e-8


def compute_batch_psnr(outputs: torch.Tensor, targets: torch.Tensor, max_val: float = 2.0) -> float:
    """Compute PSNR for a batch of reconstructed images."""
    mse = torch.mean((outputs - targets) ** 2, dim=(1, 2, 3)) + _EPS
    psnr = 20 * torch.log10(torch.tensor(max_val, device=outputs.device)) - 10 * torch.log10(mse)
    return float(psnr.mean().item())


def summarize_metrics(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> str:
    return (
        f"train_loss={train_metrics['loss']:.4f} train_psnr={train_metrics['psnr']:.2f}dB "
        f"val_loss={val_metrics['loss']:.4f} val_psnr={val_metrics['psnr']:.2f}dB"
    )


def format_time(seconds: float) -> str:
    minutes = math.floor(seconds / 60)
    secs = seconds - minutes * 60
    return f"{minutes:d}m {secs:.1f}s"
