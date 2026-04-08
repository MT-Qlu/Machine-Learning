"""Training pipeline utilities for the ARIMA forecaster."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .config import CONFIG, ARIMAConfig
from .data import load_series, train_test_split


class AirPassengersARIMAPipeline:
    """Compose ARIMA training, evaluation, and persistence."""

    def __init__(self, config: ARIMAConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model = None
        self.series_: pd.Series | None = None
        self.metrics_: dict[str, float] | None = None

    def train(self) -> dict[str, float]:
        train_series, test_series = train_test_split(self.config)
        model = ARIMA(train_series, order=self.config.order)
        results = model.fit()
        forecast = results.forecast(steps=len(test_series))

        metrics = self._evaluate(test_series, forecast)

        # Refit on the full dataset for downstream inference
        full_series = load_series(self.config)
        final_model = ARIMA(full_series, order=self.config.order).fit()

        self.model = final_model
        self.series_ = full_series
        self.metrics_ = metrics
        return metrics

    def forecast(self, horizon: int) -> pd.Series:
        if self.model is None or self.series_ is None:
            raise RuntimeError("Model not trained. Call train() first or load from disk.")
        forecast_index = pd.date_range(
            start=self.series_.index[-1] + self.series_.index.freq,
            periods=horizon,
            freq=self.series_.index.freq,
        )
        forecast_values = self.model.forecast(steps=horizon)
        return pd.Series(forecast_values, index=forecast_index)

    def save(self) -> Path:
        if self.model is None or self.series_ is None:
            raise RuntimeError("Pipeline not trained; call train() before save().")
        payload = {
            "config": self.config,
            "model": self.model,
            "series_index": self.series_.index,
            "series_values": self.series_.values,
            "metrics": self.metrics_ or {},
        }
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> "AirPassengersARIMAPipeline":
        pipeline_path = path or CONFIG.model_path
        payload = joblib.load(pipeline_path)
        pipeline = AirPassengersARIMAPipeline(payload["config"])
        pipeline.model = payload["model"]
        pipeline.series_ = pd.Series(payload["series_values"], index=payload["series_index"])
        pipeline.metrics_ = payload.get("metrics", {})
        return pipeline

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path

    @staticmethod
    def _evaluate(actual: pd.Series, forecast: pd.Series | np.ndarray) -> dict[str, float]:
        actual_values = np.asarray(actual, dtype=float)
        forecast_values = np.asarray(forecast, dtype=float)
        mae = float(np.mean(np.abs(actual_values - forecast_values)))
        rmse = float(np.sqrt(np.mean((actual_values - forecast_values) ** 2)))
        mape = float(np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}


def train_and_persist(config: ARIMAConfig | None = None) -> dict[str, float]:
    pipeline = AirPassengersARIMAPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
