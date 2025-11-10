"""Inference helpers for Gaussian Mixture Models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .pipeline import build_pipeline


class GaussianMixtureWrapper:
    """Provide convenience methods for cluster prediction and scoring."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = joblib.load(model_path) if model_path else build_pipeline

    def fit(self, features: np.ndarray) -> "GaussianMixtureWrapper":
        if callable(self.model):
            self.model = self.model()
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        return self.model.score_samples(features)


def load_model(model_path: Path):
    return joblib.load(model_path)
