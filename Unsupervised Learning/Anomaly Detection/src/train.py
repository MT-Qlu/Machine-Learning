"""Training loop scaffolding for anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from .config import AnomalyDetectionConfig
from .data import load_dataset
from .pipeline import build_pipeline


def train(config: AnomalyDetectionConfig, data_path: Path, *, artefact_dir: Optional[Path] = None) -> Path:
    """Train an anomaly detection model and persist it to the artefact directory."""

    features, _ = load_dataset(data_path)
    model = build_pipeline(config)
    model.fit(features)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "anomaly_detector.joblib"
    joblib.dump(model, model_path)
    return model_path
