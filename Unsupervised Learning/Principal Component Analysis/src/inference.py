"""Inference helpers for PCA."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .pipeline import build_pipeline


class PCATransformer:
    """Provide convenience wrappers for PCA transforms."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = joblib.load(model_path) if model_path else build_pipeline

    def fit(self, features: np.ndarray) -> "PCATransformer":
        if callable(self.model):
            self.model = self.model()
        self.model.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.inverse_transform(features)


def load_model(model_path: Path):
    return joblib.load(model_path)
