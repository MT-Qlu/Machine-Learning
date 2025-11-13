````markdown
# Variational Autoencoder

Probabilistic autoencoder that pairs an encoder producing Gaussian parameters with a decoder capable of sampling new digits. Both PyTorch and TensorFlow implementations follow the same layout as the other autoencoder variants.

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

````