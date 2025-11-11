"""Utility helpers for the TensorFlow autoencoder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf


def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.psnr(y_true, y_pred, max_val=2.0)


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            psnr_metric,
        ],
    )
    return model


def save_history(history: tf.keras.callbacks.History, summary: Dict[str, Any], path: Path) -> None:
    payload = {
        "history": history.history,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
