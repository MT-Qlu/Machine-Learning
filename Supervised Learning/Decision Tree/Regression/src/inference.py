"""Inference utilities for the California housing decision tree regressor."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, DecisionTreeRegressionConfig
from .pipeline import CaliforniaDecisionTreePipeline, train_and_persist


class CaliforniaHousingRequest(BaseModel):
    """Input schema aligned with the California housing feature space."""

    median_income: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0)
    average_rooms: float = Field(..., gt=0)
    average_bedrooms: float = Field(..., gt=0)
    population: float = Field(..., ge=0)
    average_occupancy: float = Field(..., gt=0)
    latitude: float = Field(..., ge=32, le=43)
    longitude: float = Field(..., ge=-125, le=-114)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "median_income": 6.5,
                "house_age": 28.0,
                "average_rooms": 5.4,
                "average_bedrooms": 1.1,
                "population": 830.0,
                "average_occupancy": 3.2,
                "latitude": 34.1,
                "longitude": -118.2,
            }
        }
    )


class CaliforniaHousingResponse(BaseModel):
    """Prediction payload served by the FastAPI endpoint."""

    predicted_value: float
    feature_importances: dict[str, float]
    model_version: str
    metrics: dict[str, float]


class CaliforniaHousingService:
    """High-level service object for California housing value predictions."""

    def __init__(self, config: DecisionTreeRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return CaliforniaDecisionTreePipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: CaliforniaHousingRequest) -> CaliforniaHousingResponse:
        record = {column: getattr(payload, column) for column in self.config.feature_columns}
        data = pd.DataFrame([record])
        prediction = float(self.pipeline.predict(data)[0])
        regressor = self.pipeline.named_steps["regressor"]
        feature_importances = {
            feature: float(importance)
            for feature, importance in zip(self.config.feature_columns, regressor.feature_importances_)
        }
        model_version = self._artifact_version(self.config.model_path)
        return CaliforniaHousingResponse(
            predicted_value=prediction,
            feature_importances=feature_importances,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> CaliforniaHousingService:
    return CaliforniaHousingService()


RequestModel = CaliforniaHousingRequest
ResponseModel = CaliforniaHousingResponse
