"""Inference helpers for K-Means clustering."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .pipeline import build_pipeline


class KMeansClusterer:
    """Wrap K-Means to expose ``fit``/``predict`` helpers consistent with supervised modules."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = joblib.load(model_path) if model_path else build_pipeline

    def fit(self, features: np.ndarray) -> "KMeansClusterer":
        if callable(self.model):
            self.model = self.model()
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)


def load_model(model_path: Path):
    return joblib.load(model_path)
