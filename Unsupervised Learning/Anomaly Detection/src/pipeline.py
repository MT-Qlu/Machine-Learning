"""Pipeline constructors for anomaly detection models."""

from __future__ import annotations

from sklearn.ensemble import IsolationForest

from .config import AnomalyDetectionConfig


def build_pipeline(config: AnomalyDetectionConfig) -> IsolationForest:
    """Return an IsolationForest configured for anomaly detection.

    This thin wrapper mirrors the supervised modules and offers a single place
    to extend preprocessing or model choices later (e.g. add LOF or autoencoders).
    """

    return IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
    )
