"""Inference helpers for the TensorFlow deconvolutional autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CONFIG, DeconvConfig
from .model import build_model
from .utils import compile_model, psnr_metric


def load_model(weights_path: Path | None = None, config: DeconvConfig = CONFIG) -> tf.keras.Model:
    path = weights_path or config.artifact_dir / "deconv_autoencoder.h5"
    custom_objects = {"psnr_metric": psnr_metric}
    if path.exists():
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    else:
        model = build_model()
        model = compile_model(model, learning_rate=config.learning_rate)
    return model


def reconstruct(inputs: Iterable[np.ndarray], model: tf.keras.Model | None = None) -> List[np.ndarray]:
    model = model or load_model()
    outputs: List[np.ndarray] = []
    for array in inputs:
        batch = np.expand_dims(array, axis=0)
        recon = model.predict(batch, verbose=0)
        outputs.append(recon.squeeze(0))
    return outputs
