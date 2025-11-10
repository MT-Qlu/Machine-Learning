"""Inference helpers for DBSCAN clustering."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .pipeline import build_pipeline


class DBSCANClusterer:
    """Wraps a DBSCAN model with simple ``fit``/``predict`` helpers."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = joblib.load(model_path) if model_path else build_pipeline

    def fit(self, features: np.ndarray) -> "DBSCANClusterer":
        if callable(self.model):
            self.model = self.model()
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(features)


def load_model(model_path: Path):
    return joblib.load(model_path)
