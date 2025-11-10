"""Training scaffolding for PCA."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import PCAConfig
from .data import load_features
from .pipeline import build_pipeline


def train(config: PCAConfig, data_path: Path, *, artefact_dir: Path | None = None) -> Path:
    """Fit PCA and persist the transformer."""

    features = load_features(data_path)
    transformer = build_pipeline(config)
    transformer.fit(features)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "pca.joblib"
    joblib.dump(transformer, model_path)
    return model_path
