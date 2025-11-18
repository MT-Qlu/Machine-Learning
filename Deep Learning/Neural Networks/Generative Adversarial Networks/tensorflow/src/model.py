import tensorflow as tf


def build_generator(latent_dim: int = 100, channels: int = 1, feature_maps: int = 64) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * feature_maps * 4, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((7, 7, feature_maps * 4))(x)

    x = tf.keras.layers.Conv2DTranspose(feature_maps * 2, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(feature_maps, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same", activation="tanh")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="generator")


def build_discriminator(channels: int = 1, feature_maps: int = 64) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(28, 28, channels))

    x = tf.keras.layers.Conv2D(feature_maps, 4, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(feature_maps * 2, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(feature_maps * 4, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="discriminator")
