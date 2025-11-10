"""Training scaffolding for Independent Component Analysis."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import ICAConfig
from .data import load_features
from .pipeline import build_pipeline


def train(config: ICAConfig, data_path: Path, *, artefact_dir: Path | None = None) -> Path:
    """Fit FastICA on the dataset and persist the unmixing matrix."""

    features = load_features(data_path)
    transformer = build_pipeline(config)
    transformed = transformer.fit_transform(features)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "ica.joblib"
    components_path = artefact_dir / "ica_components.npy"
    joblib.dump(transformer, model_path)
    joblib.dump(transformed, components_path)
    return model_path
