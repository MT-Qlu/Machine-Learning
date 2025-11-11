"""Inference helpers for the trained PyTorch CNN."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch

from .config import CNNConfig, CONFIG
from .model import FashionMNISTCNN


def load_model(weights_path: Path | None = None, config: CNNConfig = CONFIG) -> FashionMNISTCNN:
    path = weights_path or config.artifact_dir / "cnn_pytorch.pt"
    model = FashionMNISTCNN()
    state_dict = torch.load(path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def predict(inputs: Iterable[torch.Tensor], model: FashionMNISTCNN | None = None, config: CNNConfig = CONFIG) -> List[int]:
    model = model or load_model(config=config)
    predictions: List[int] = []
    with torch.no_grad():
        for tensor in inputs:
            tensor = tensor.to(config.device)
            logits = model(tensor.unsqueeze(0))
            predictions.append(int(logits.argmax(dim=1).item()))
    return predictions
