"""Inference utilities for serving One-vs-One predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, OneVsOneConfig
from .pipeline import OneVsOnePipeline


class OneVsOneRequest(BaseModel):
    """Request schema for iris classification via One-vs-One."""

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


class OneVsOneResponse(BaseModel):
    """Response schema returned by inference calls."""

    predicted_class: int
    predicted_class_name: str
    confidence_scores: list[float]
    n_binary_classifiers: int
    model_version: str

    model_config = ConfigDict(use_enum_values=True)


class OneVsOneService:
    """High-level service object that wraps the trained One-vs-One pipeline."""

    IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    def __init__(self, config: OneVsOneConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            pipeline = OneVsOnePipeline(self.config)
            pipeline.train()
        else:
            pipeline = OneVsOnePipeline(self.config)
            # Load trained model
            models = joblib.load(self.config.model_path)
            pipeline.ovo_classifier = models["ovo_classifier"]
            pipeline.scaler = models["scaler"]
        return pipeline

    def predict(self, payload: OneVsOneRequest) -> OneVsOneResponse:
        """Generate prediction using One-vs-One strategy."""
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
            models = joblib.load(self.config.model_path)
            scaler = models["scaler"]
            features_scaled = scaler.transform(features)

        # Predict
        y_pred = self.pipeline.ovo_classifier.predict(features_scaled)[0]
        confidence = self.pipeline.ovo_classifier.predict_proba(features_scaled)[0]

        # Calculate number of binary classifiers: K(K-1)/2
        n_classes = len(self.IRIS_CLASSES)
        n_estimators = n_classes * (n_classes - 1) // 2

        return OneVsOneResponse(
            predicted_class=int(y_pred),
            predicted_class_name=self.IRIS_CLASSES[int(y_pred)],
            confidence_scores=[float(c) for c in confidence],
            n_binary_classifiers=n_estimators,
            model_version=self._artifact_version(self.config.model_path),
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> OneVsOneService:
    """Factory returning a cached service instance for FastAPI integration."""
    return OneVsOneService()


RequestModel = OneVsOneRequest
ResponseModel = OneVsOneResponse
