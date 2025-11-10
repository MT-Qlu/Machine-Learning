"""Inference helper mirroring the supervised modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .pipeline import build_pipeline


class AnomalyDetector:
    """Thin wrapper that exposes ``predict`` and ``score_samples`` helpers."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = joblib.load(model_path) if model_path else build_pipeline

    def fit(self, features: np.ndarray) -> "AnomalyDetector":
        if callable(self.model):  # handle lazy construction when no artefact was supplied
            self.model = self.model()
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        return self.model.score_samples(features)


def load_model(model_path: Path) -> Any:
    """Load a persisted anomaly detector."""

    return joblib.load(model_path)
