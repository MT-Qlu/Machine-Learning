"""Inference utilities for serving One-vs-Rest predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, OneVsRestConfig
from .pipeline import OneVsRestPipeline


class OneVsRestRequest(BaseModel):
    """Request schema for iris classification via One-vs-Rest."""

    sepal_length: float = Field(..., gt=0.0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0.0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0.0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0.0, description="Petal width in cm")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )


class OneVsRestResponse(BaseModel):
    """Response schema returned by inference calls."""

    predicted_class: int
    predicted_class_name: str
    confidence_scores: list[float]
    n_binary_classifiers: int
    model_version: str

    model_config = ConfigDict(use_enum_values=True)


class OneVsRestService:
    """High-level service object that wraps the trained One-vs-Rest pipeline."""

    IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    def __init__(self, config: OneVsRestConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            pipeline = OneVsRestPipeline(self.config)
            pipeline.train()
        else:
            pipeline = OneVsRestPipeline(self.config)
            # Load trained model
            models = joblib.load(self.config.model_path)
            pipeline.ovr_classifier = models["ovr_classifier"]
            pipeline.scaler = models["scaler"]
        return pipeline

    def predict(self, payload: OneVsRestRequest) -> OneVsRestResponse:
        """Generate prediction using One-vs-Rest strategy."""
        features = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]], dtype=float)

        # Scale features
        if hasattr(self.pipeline, "scaler"):
            features_scaled = self.pipeline.scaler.transform(features)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            models = joblib.load(self.config.model_path)
            scaler = models["scaler"]
            features_scaled = scaler.transform(features)

        # Predict
        y_pred = self.pipeline.ovr_classifier.predict(features_scaled)[0]
        confidence = self.pipeline.ovr_classifier.predict_proba(features_scaled)[0]

        return OneVsRestResponse(
            predicted_class=int(y_pred),
            predicted_class_name=self.IRIS_CLASSES[int(y_pred)],
            confidence_scores=[float(c) for c in confidence],
            n_binary_classifiers=self.pipeline.ovr_classifier.n_classes_,
            model_version=self._artifact_version(self.config.model_path),
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> OneVsRestService:
    """Factory returning a cached service instance for FastAPI integration."""
    return OneVsRestService()


RequestModel = OneVsRestRequest
ResponseModel = OneVsRestResponse
