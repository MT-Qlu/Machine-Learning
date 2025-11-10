"""Training scaffolding for Gaussian Mixture Models."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import GaussianMixtureConfig
from .data import load_features
from .pipeline import build_pipeline


def train(config: GaussianMixtureConfig, data_path: Path, *, artefact_dir: Path | None = None) -> Path:
    """Fit a GaussianMixture model and persist it to the artefact directory."""

    features = load_features(data_path)
    model = build_pipeline(config)
    model.fit(features)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "gaussian_mixture.joblib"
    joblib.dump(model, model_path)
    return model_path
