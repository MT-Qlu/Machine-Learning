from __future__ import annotations

import tensorflow as tf

from .config import GANConfig


class GANEngine:
    def __init__(self, config: GANConfig) -> None:
        self.config = config
        self.latent_dim = config.latent_dim
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def sample_noise(self, batch_size: int) -> tf.Tensor:
        return tf.random.normal((batch_size, self.latent_dim))

    def discriminator_step(
        self,
        discriminator: tf.keras.Model,
        generator: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        real_images: tf.Tensor,
    ) -> float:
        batch_size = tf.shape(real_images)[0]
        noise = self.sample_noise(batch_size)

        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)

            real_loss = self.bce(tf.ones_like(real_logits), real_logits)
            fake_loss = self.bce(tf.zeros_like(fake_logits), fake_logits)
            loss = real_loss + fake_loss

        grads = tape.gradient(loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return float(loss.numpy())

    def generator_step(
        self,
        discriminator: tf.keras.Model,
        generator: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        batch_size: int,
    ) -> float:
        noise = self.sample_noise(batch_size)

        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            fake_logits = discriminator(fake_images, training=True)
            loss = self.bce(tf.ones_like(fake_logits), fake_logits)

        grads = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return float(loss.numpy())
