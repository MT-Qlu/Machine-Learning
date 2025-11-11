"""Scriptable training entry point for the deconvolutional autoencoder."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW

from .config import CONFIG, DeconvConfig
from .data import create_dataloaders
from .engine import evaluate, train_one_epoch
from .model import DeconvAutoencoder
from .utils import format_time, summarize_metrics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]


def train(config: DeconvConfig = CONFIG) -> Dict[str, float]:
    config.ensure_dirs()
    set_seed(config.seed)

    train_loader, val_loader = create_dataloaders(config)
    device = config.device

    model = DeconvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        print(f"Epoch {epoch}/{config.num_epochs} - {summarize_metrics(train_metrics, val_metrics)}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(model, config.artifact_dir / "deconv_autoencoder.pt", config)

    duration = time.time() - start_time
    final_metrics = history["val"][-1]
    summary = {
        "best_val_loss": best_val_loss,
        "best_val_psnr": max(m["psnr"] for m in history["val"]),
        "final_val_loss": final_metrics["loss"],
        "final_val_psnr": final_metrics["psnr"],
        "train_duration": format_time(duration),
        "device": str(device),
    }

    _save_metrics(history, summary, config.artifact_dir / "metrics.json")
    return summary


def _save_checkpoint(model: nn.Module, path: Path, config: DeconvConfig) -> None:
    config.ensure_dirs()
    torch.save(model.state_dict(), path)


def _save_metrics(history: Dict[str, List[Dict[str, float]]], summary: Dict[str, float], path: Path) -> None:
    payload = {"history": history, "summary": summary}
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
