"""Inference utilities for serving voting ensemble predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, VotingConfig
from .pipeline import VotingEnsemblePipeline


class VotingEnsembleRequest(BaseModel):
    """Request schema for iris classification via voting ensemble."""

    sepal_length: float = Field(..., gt=0.0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0.0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0.0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0.0, description="Petal width in cm")
    voting_method: str = Field(default="soft", description="'hard' or 'soft' voting")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "voting_method": "soft",
            }
        }
    )


class VotingEnsembleResponse(BaseModel):
    """Response schema returned by inference calls."""

    predicted_class: int
    predicted_class_name: str
    confidence_scores: list[float]
    voting_method: str
    model_version: str

    model_config = ConfigDict(use_enum_values=True)


class VotingEnsembleService:
    """High-level service object that wraps the trained voting ensemble pipeline."""

    IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    def __init__(self, config: VotingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            pipeline = VotingEnsemblePipeline(self.config)
            pipeline.train()
        else:
            pipeline = VotingEnsemblePipeline(self.config)
            # Load trained models
            # Note: Voting models are stored in joblib format
        return pipeline

    def predict(self, payload: VotingEnsembleRequest) -> VotingEnsembleResponse:
        """Generate prediction using voting ensemble."""
        features = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]], dtype=float)

        # Select voting method
        if payload.voting_method == "hard":
            if self.pipeline.hard_voting is None:
                self.pipeline.train()
            y_pred = self.pipeline.hard_voting.predict(features)[0]
            confidence = self.pipeline.hard_voting.predict_proba(features)[0]
        else:  # soft
            if self.pipeline.soft_voting is None:
                self.pipeline.train()
            y_pred = self.pipeline.soft_voting.predict(features)[0]
            confidence = self.pipeline.soft_voting.predict_proba(features)[0]

        return VotingEnsembleResponse(
            predicted_class=int(y_pred),
            predicted_class_name=self.IRIS_CLASSES[int(y_pred)],
            confidence_scores=[float(c) for c in confidence],
            voting_method=payload.voting_method,
            model_version=self._artifact_version(),
        )

    @staticmethod
    def _artifact_version() -> str:
        return "1.0"


@lru_cache(maxsize=1)
def get_service() -> VotingEnsembleService:
    """Factory returning a cached service instance for FastAPI integration."""
    return VotingEnsembleService()


RequestModel = VotingEnsembleRequest
ResponseModel = VotingEnsembleResponse
