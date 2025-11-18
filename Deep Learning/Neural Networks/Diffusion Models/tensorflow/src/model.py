from __future__ import annotations

import math
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SinusoidalTimeEmbedding(layers.Layer):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, timesteps: tf.Tensor) -> tf.Tensor:
        half_dim = self.dim // 2
        emb = tf.cast(tf.range(half_dim), tf.float32)
        emb = tf.exp(-math.log(10000.0) * emb / tf.cast(half_dim - 1, tf.float32))
        emb = tf.cast(tf.expand_dims(timesteps, -1), tf.float32) * emb
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb


def residual_block(x: tf.Tensor, t_emb: tf.Tensor, out_channels: int, name: str) -> tf.Tensor:
    input_channels = x.shape[-1]

    h = layers.GroupNormalization(groups=8, name=f"{name}_gn1")(x)
    h = layers.Activation("swish", name=f"{name}_act1")(h)
    h = layers.Conv2D(out_channels, kernel_size=3, padding="same", name=f"{name}_conv1")(h)

    t_proj = layers.Dense(out_channels, name=f"{name}_time_dense")(t_emb)
    t_proj = layers.Activation("swish", name=f"{name}_time_act")(t_proj)
    t_proj = layers.Reshape((1, 1, out_channels), name=f"{name}_time_reshape")(t_proj)

    h = layers.Add(name=f"{name}_add_time")([h, t_proj])
    h = layers.GroupNormalization(groups=8, name=f"{name}_gn2")(h)
    h = layers.Activation("swish", name=f"{name}_act2")(h)
    h = layers.Conv2D(out_channels, kernel_size=3, padding="same", name=f"{name}_conv2")(h)

    shortcut = x
    if input_channels != out_channels:
        shortcut = layers.Conv2D(out_channels, kernel_size=1, padding="same", name=f"{name}_shortcut")(shortcut)

    return layers.Add(name=f"{name}_add")([h, shortcut])


def build_unet(image_size: int = 28, channels: int = 1, time_dim: int = 128, base_channels: int = 64) -> keras.Model:
    inputs = layers.Input(shape=(image_size, image_size, channels), name="x")
    t_in = layers.Input(shape=(), dtype=tf.int32, name="timesteps")

    t_emb = SinusoidalTimeEmbedding(time_dim, name="time_embedding")(t_in)
    t_emb = layers.Dense(time_dim * 4, activation="swish", name="time_dense1")(t_emb)
    t_emb = layers.Dense(time_dim, activation="swish", name="time_dense2")(t_emb)

    x = layers.Conv2D(base_channels, kernel_size=3, padding="same", name="initial_conv")(inputs)

    d1 = residual_block(x, t_emb, base_channels * 2, name="down_block1")
    x_down1 = layers.Conv2D(base_channels * 2, kernel_size=3, strides=2, padding="same", name="downsample1")(d1)

    d2 = residual_block(x_down1, t_emb, base_channels * 4, name="down_block2")
    x_down2 = layers.Conv2D(base_channels * 4, kernel_size=3, strides=2, padding="same", name="downsample2")(d2)

    mid = residual_block(x_down2, t_emb, base_channels * 4, name="mid_block1")
    mid = residual_block(mid, t_emb, base_channels * 4, name="mid_block2")

    u1 = layers.Conv2DTranspose(base_channels * 2, kernel_size=4, strides=2, padding="same", name="upsample1")(mid)
    u1 = layers.Concatenate(name="concat1")([u1, d2])
    u1 = residual_block(u1, t_emb, base_channels * 2, name="up_block1")

    u2 = layers.Conv2DTranspose(base_channels, kernel_size=4, strides=2, padding="same", name="upsample2")(u1)
    u2 = layers.Concatenate(name="concat2")([u2, d1])
    u2 = residual_block(u2, t_emb, base_channels, name="up_block2")

    out = layers.GroupNormalization(groups=8, name="final_gn")(u2)
    out = layers.Activation("swish", name="final_act")(out)
    out = layers.Conv2D(channels, kernel_size=3, padding="same", name="final_conv")(out)

    return keras.Model(inputs=[inputs, t_in], outputs=out, name="ddpm_unet")
