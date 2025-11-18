from __future__ import annotations

import tensorflow as tf

from .config import DiffusionConfig
from .utils import extract, linear_beta_schedule


class DiffusionEngine:
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.device = config.strategy().scope()
        self.betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = tf.math.cumprod(self.alphas, axis=0)

    def sample_timesteps(self, batch_size: int) -> tf.Tensor:
        return tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.config.timesteps, dtype=tf.int32)

    def q_sample(self, x_start: tf.Tensor, t: tf.Tensor, noise: tf.Tensor | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_start))
        sqrt_alpha_hat = tf.sqrt(extract(self.alpha_hats, t, x_start.shape))
        sqrt_one_minus_alpha_hat = tf.sqrt(1.0 - extract(self.alpha_hats, t, x_start.shape))
        x_t = sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def loss(self, model: tf.keras.Model, x_start: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        x_t, noise = self.q_sample(x_start, t)
        pred_noise = model([x_t, t], training=True)
        return tf.reduce_mean(tf.square(pred_noise - noise))

    def sample(self, model: tf.keras.Model, num_images: int) -> tf.Tensor:
        images = tf.random.normal(shape=(num_images, self.config.image_size, self.config.image_size, self.config.channels))

        step_indices = tf.cast(
            tf.linspace(0, self.config.timesteps - 1, self.config.sample_steps),
            tf.int32,
        )

        for idx in reversed(step_indices):
            t = tf.fill([num_images], idx)
            beta_t = tf.gather(self.betas, t)
            alpha_t = tf.gather(self.alphas, t)
            alpha_hat_t = tf.gather(self.alpha_hats, t)

            pred_noise = model([images, t], training=False)

            noise = tf.where(
                tf.equal(idx, 0),
                tf.zeros_like(images),
                tf.random.normal(shape=tf.shape(images)),
            )

            images = (
                1.0 / tf.sqrt(alpha_t)[:, None, None, None]
                * (images - beta_t[:, None, None, None] / tf.sqrt(1.0 - alpha_hat_t)[:, None, None, None] * pred_noise)
                + tf.sqrt(beta_t)[:, None, None, None] * noise
            )

        return images
