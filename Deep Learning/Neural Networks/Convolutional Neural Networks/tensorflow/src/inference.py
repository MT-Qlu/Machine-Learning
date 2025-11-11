"""Inference helpers for the TensorFlow CNN."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CNNConfig, CONFIG
from .model import build_model


def load_model(weights_path: Path | None = None, config: CNNConfig = CONFIG) -> tf.keras.Model:
    path = weights_path or config.artifact_dir / "cnn_tensorflow.h5"
    if path.exists():
        model = tf.keras.models.load_model(path)
    else:
        model = build_model()
    return model


def predict(inputs: Iterable[np.ndarray], model: tf.keras.Model | None = None) -> List[int]:
    model = model or build_model()
    predictions: List[int] = []
    for array in inputs:
        logits = model.predict(np.expand_dims(array, axis=0), verbose=0)
        predictions.append(int(np.argmax(logits, axis=1)[0]))
    return predictions
