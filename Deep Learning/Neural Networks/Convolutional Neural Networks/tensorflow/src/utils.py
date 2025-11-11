"""Utility helpers for TensorFlow CNN workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import tensorflow as tf


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
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
