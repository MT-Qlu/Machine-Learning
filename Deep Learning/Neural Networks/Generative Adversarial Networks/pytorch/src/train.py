from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import optim
from tqdm.auto import tqdm

from .config import CONFIG, GANConfig
from .data import load_dataloaders
from .engine import GANEngine
from .model import Discriminator, Generator
from .utils import save_image_grid, seed_everything, write_metrics


def _ensure_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train(config: GANConfig = CONFIG) -> Dict[str, list[float]]:
    seed_everything(config.seed)
    device = config.device
    _ensure_artifacts(config.artifact_dir)

    dataloader, dataset_size = load_dataloaders(config)

    generator = Generator(latent_dim=config.latent_dim, channels=config.channels).to(device)
    discriminator = Discriminator(channels=config.channels).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config.g_lr, betas=config.betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=config.betas)

    engine = GANEngine(config)
    metrics: Dict[str, list[float]] = {"g_loss": [], "d_loss": []}

    for epoch in range(config.epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for real_images, _ in progress:
            real_images = real_images.to(device)
            d_loss = engine.discriminator_step(discriminator, generator, optimizer_d, real_images)
            g_loss = engine.generator_step(discriminator, generator, optimizer_g, real_images.size(0))
            d_epoch_loss += d_loss * real_images.size(0)
            g_epoch_loss += g_loss * real_images.size(0)
            progress.set_postfix(d_loss=d_loss, g_loss=g_loss)

        metrics["d_loss"].append(d_epoch_loss / dataset_size)
        metrics["g_loss"].append(g_epoch_loss / dataset_size)

    torch.save(generator.state_dict(), config.generator_ckpt)
    torch.save(discriminator.state_dict(), config.discriminator_ckpt)

    with torch.no_grad():
        fixed_noise = engine.sample_noise(config.sample_batch_size)
        samples = generator(fixed_noise).cpu()
        save_image_grid(samples, config.grid_path, nrow=8)

    write_metrics(metrics, config.metrics_file)
    return metrics
