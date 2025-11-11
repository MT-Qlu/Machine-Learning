"""Inference helpers for the deconvolutional autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch

from .config import CONFIG, DeconvConfig
from .model import DeconvAutoencoder


def load_model(weights_path: Path | None = None, config: DeconvConfig = CONFIG) -> DeconvAutoencoder:
    path = weights_path or config.artifact_dir / "deconv_autoencoder.pt"
    model = DeconvAutoencoder()
    if path.exists():
        state_dict = torch.load(path, map_location=config.device)
        model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def reconstruct(
    inputs: Iterable[torch.Tensor],
    model: DeconvAutoencoder | None = None,
    config: DeconvConfig = CONFIG,
) -> List[torch.Tensor]:
    model = model or load_model(config=config)
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for tensor in inputs:
            tensor = tensor.to(config.device)
            recon = model(tensor.unsqueeze(0))
            outputs.append(recon.squeeze(0).cpu())
    return outputs
