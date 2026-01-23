# Variational Autoencoder

Probabilistic autoencoder that pairs an encoder producing Gaussian parameters with a decoder capable of sampling new digits. Both PyTorch and TensorFlow implementations follow the same layout as the other autoencoder variants.

---

## Theory primer

VAEs model a **distribution** over latent variables. The encoder predicts parameters of a Gaussian posterior $q_\theta(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mu, \sigma^2 I)$, and the decoder defines $p_\phi(\mathbf{x}\mid\mathbf{z})$.

The loss is the negative ELBO:

$$
\mathcal{L} = \mathbb{E}_{q_\theta(\mathbf{z}\mid\mathbf{x})}\big[\|\mathbf{x} - g_\phi(\mathbf{z})\|_2^2\big] + \text{KL}(q_\theta(\mathbf{z}\mid\mathbf{x})\;\|\;p(\mathbf{z}))
$$

with a standard normal prior $p(\mathbf{z}) = \mathcal{N}(0, I)$. The **reparameterisation trick** enables backpropagation through sampling:

$$
\mathbf{z} = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Balancing reconstruction and KL terms controls **sample quality vs. latent smoothness**.

- `pytorch/` — Torch VAE with KL tracking and sampling helpers.
- `tensorflow/` — Keras VAE with custom training step and notebook tour.

Use the provided notebooks to compare KL annealing schedules or visualise interpolations in latent space.

---

## Learning goals

- Study how KL regularisation shapes smooth latent manifolds suitable for sampling.
- Evaluate reconstruction vs KL trade-offs under different annealing schedules.
- Practice generating new samples and latent traversals to assess model quality.

---

## Implementation highlights

- Shared project layout ensures parity between PyTorch and TensorFlow experiments.
- Training scripts log separate KL and reconstruction losses for deep-dive analysis.
- Inference utilities expose sampling APIs so you can script custom visualisations quickly.