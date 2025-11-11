"""Scriptable training for the TensorFlow CNN."""
from __future__ import annotations

import json
from typing import Dict

import tensorflow as tf

from .config import CNNConfig, CONFIG
from .data import load_datasets
from .model import build_model
from .utils import compile_model, save_history


def set_seed(seed: int) -> None:
    tf.random.set_seed(seed)


def train(config: CNNConfig = CONFIG) -> Dict[str, float]:
    config.ensure_dirs()
    set_seed(config.seed)

    train_ds, val_ds = load_datasets(config)
    model = build_model()
    model = compile_model(model, learning_rate=config.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.artifact_dir / "cnn_tensorflow.h5"),
            monitor="val_accuracy",
            save_best_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.num_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    final_metrics = {
        "final_val_accuracy": history.history.get("val_accuracy", [0.0])[-1],
        "final_val_loss": history.history.get("val_loss", [0.0])[-1],
        "best_val_accuracy": best_val_acc,
    }

    save_history(history, final_metrics, config.artifact_dir / "metrics.json")
    return final_metrics


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
