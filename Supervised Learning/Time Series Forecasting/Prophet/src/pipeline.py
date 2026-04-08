"""Training pipeline utilities for the Prophet forecaster."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from .config import CONFIG, ProphetConfig
from .data import load_dataframe, train_test_split


class AirPassengersProphetPipeline:
    """Compose Prophet training, evaluation, and persistence."""

    def __init__(self, config: ProphetConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model: Prophet | None = None
        self.history_: pd.DataFrame | None = None
        self.metrics_: dict[str, float] | None = None

    def train(self) -> dict[str, float]:
        train_df, test_df = train_test_split(self.config)
        model = Prophet(
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test_df), freq=self.config.frequency)
        forecast_df = model.predict(future)
        forecast_tail = forecast_df.tail(len(test_df))

        metrics = self._evaluate(test_df[self.config.target_column].values, forecast_tail["yhat"].values)

        # Refit on full history for deployment
        full_df = load_dataframe(self.config)
        final_model = Prophet(
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        final_model.fit(full_df)

        self.model = final_model
        self.history_ = full_df
        self.metrics_ = metrics
        return metrics

    def forecast(self, horizon: int) -> pd.DataFrame:
        if self.model is None or self.history_ is None:
            raise RuntimeError("Model not trained. Call train() first or load from disk.")
        future = self.model.make_future_dataframe(periods=horizon, freq=self.config.frequency)
        forecast_df = self.model.predict(future)
        return forecast_df.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def save(self) -> Path:
        if self.model is None or self.history_ is None:
            raise RuntimeError("Pipeline not trained; call train() before save().")
        payload = {
            "config": self.config,
            "model": self.model,
            "history": self.history_,
            "metrics": self.metrics_ or {},
        }
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> "AirPassengersProphetPipeline":
        pipeline_path = path or CONFIG.model_path
        payload = joblib.load(pipeline_path)
        pipeline = AirPassengersProphetPipeline(payload["config"])
        pipeline.model = payload["model"]
        pipeline.history_ = payload["history"]
        pipeline.metrics_ = payload.get("metrics", {})
        return pipeline

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path

    @staticmethod
    def _evaluate(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
        mae = float(np.mean(np.abs(actual - forecast)))
        rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
        mape = float(np.mean(np.abs((actual - forecast) / actual)) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}


def train_and_persist(config: ProphetConfig | None = None) -> dict[str, float]:
    pipeline = AirPassengersProphetPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
