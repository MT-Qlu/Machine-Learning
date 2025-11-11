"""Scriptable training entry point for the PyTorch CNN."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW

from .config import CNNConfig, CONFIG
from .data import create_dataloaders
from .engine import evaluate, train_one_epoch
from .model import FashionMNISTCNN
from .utils import format_time, summarize_metrics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]


def train(config: CNNConfig = CONFIG) -> Dict[str, float]:
    """Train the CNN and persist metrics + weights."""
    config.ensure_dirs()
    set_seed(config.seed)

    device = config.device
    train_loader, val_loader = create_dataloaders(config)

    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = {"train": [], "val": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        print(f"Epoch {epoch}/{config.num_epochs} - {summarize_metrics(train_metrics, val_metrics)}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            _save_checkpoint(model, config.artifact_dir / "cnn_pytorch.pt", config)

    duration = time.time() - start_time
    final_metrics = history["val"][-1]
    result = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": final_metrics["accuracy"],
        "final_val_loss": final_metrics["loss"],
        "train_duration": format_time(duration),
        "device": str(device),
    }

    _save_metrics(history, result, config.artifact_dir / "metrics.json")
    return result


def _save_checkpoint(model: nn.Module, path: Path, config: CNNConfig) -> None:
    config.ensure_dirs()
    torch.save(model.state_dict(), path)


def _save_metrics(history: Dict[str, list], summary: Dict[str, float], path: Path) -> None:
    payload = {"history": history, "summary": summary}
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
