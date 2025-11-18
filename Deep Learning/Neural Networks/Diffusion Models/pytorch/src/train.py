from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import optim
from tqdm.auto import tqdm

from .config import CONFIG, DiffusionConfig
from .data import load_dataloaders
from .engine import DiffusionEngine
from .model import UNet
from .utils import save_image_grid, seed_everything, write_metrics


def _ensure_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train(config: DiffusionConfig = CONFIG) -> Dict[str, list[float]]:
    """Train the DDPM on Fashion-MNIST and return logged metrics."""

    seed_everything(config.seed)
    device = config.device
    _ensure_artifacts(config.artifact_dir)

    train_loader, val_loader = load_dataloaders(config)
    model = UNet(channels=config.channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    engine = DiffusionEngine(config, device)

    metrics: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            t = engine.sample_timesteps(images.shape[0])
            loss = engine.p_losses(model, images, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        metrics["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                t = engine.sample_timesteps(images.shape[0])
                loss = engine.p_losses(model, images, t)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        metrics["val_loss"].append(val_loss)

    torch.save(model.state_dict(), config.checkpoint_file)

    samples = engine.sample(model, config.sample_batch_size, config.image_size)
    save_image_grid(samples, config.artifact_dir / "ddpm_samples.png", nrow=8)
    write_metrics(metrics, config.metrics_file)

    return metrics
