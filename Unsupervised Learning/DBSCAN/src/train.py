"""Training scaffolding for DBSCAN."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import DBSCANConfig
from .data import load_features
from .pipeline import build_pipeline


def train(config: DBSCANConfig, data_path: Path, *, artefact_dir: Path | None = None) -> Path:
    """Fit DBSCAN on the provided dataset and persist the clustering model."""

    features = load_features(data_path)
    model = build_pipeline(config)
    model.fit(features)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "dbscan.joblib"
    joblib.dump(model, model_path)
    return model_path
