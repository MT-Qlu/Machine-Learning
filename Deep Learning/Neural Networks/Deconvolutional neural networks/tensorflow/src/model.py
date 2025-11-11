"""Keras autoencoder using Conv2DTranspose layers."""
from __future__ import annotations

import tensorflow as tf


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="tanh")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="deconv_autoencoder")
    return model
