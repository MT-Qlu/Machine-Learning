# Sparse Autoencoder

Targets compact latent representations by adding a KL divergence penalty that pushes activations toward a low desired firing rate.

---

## Theory primer

Sparse autoencoders encourage most latent units to be **inactive** for any given input. A common formulation matches the average activation of each latent unit $\hat{\rho}_j$ to a small target $\rho$ using KL divergence:

$$
\hat{\rho}_j = \frac{1}{m} \sum_{i=1}^m a_j(\mathbf{x}^{(i)})
$$

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \sum_{j} \text{KL}(\rho\;\|\;\hat{\rho}_j)
$$

with

$$
\mathrm{KL}(\rho\;\|\;\hat{\rho}) = \rho \log \frac{\rho}{\hat{\rho}} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}}
$$

The penalty prevents trivial identity mappings even when the latent space is large, yielding **feature-like** encodings that resemble parts-based representations.

---

## Learning goals

- See how sparsity regularisation influences encoder behaviour and reconstruction quality.
- Compare how PyTorch and TensorFlow implement custom penalties (loss augmentation vs overridden `train_step`).
- Use logged KL metrics to diagnose under- or over-regularisation.

---

## Directory tour

- `pytorch/` — Torch pipeline with KL divergence utilities and a metrics-rich training loop.
- `tensorflow/` — Subclassed Keras model that integrates the penalty directly inside the optimisation step.

---

## Suggested experiments

1. Plot latent activation histograms across epochs to confirm the sparsity target is met.
2. Combine the sparsity penalty with denoising by importing the noisy dataset wrapper from the previous module.
3. Vary latent dimensionality to examine whether the penalty or bottleneck dominates the learning dynamics.