# Denoising Autoencoder

Strengthens the vanilla baseline by learning to map noisy Fashion-MNIST images back to their clean counterparts. Use this module to explore how explicit corruption changes the objective and the behaviour of the latent space.

---

## Theory primer

Denoising autoencoders learn to **invert a corruption process**. Given a clean input $\mathbf{x}$, a noisy version $\tilde{\mathbf{x}}$ is sampled (e.g., Gaussian noise):

$$
\widetilde{\mathbf{x}} = \mathbf{x} + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

The model is trained to reconstruct the clean input from the corrupted one:

$$
\mathcal{L}_{\text{denoise}} = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\big[\|\mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}}))\|_2^2\big]
$$

This encourages the encoder to capture **stable, noise-robust features** rather than pixel-level noise. In practice, stronger noise improves robustness but can hurt reconstruction fidelity if it overwhelms the signal.

---

## Learning goals

- Diagnose how different noise distributions and magnitudes affect reconstruction quality.
- Compare model resilience across frameworks by examining PSNR and MSE curves.
- Understand where to inject corruption in a data pipeline without changing model code.

---

## Directory tour

- `pytorch/` — Torch package with an on-the-fly noisy dataset wrapper, denoising inference helper, and guided notebook.
- `tensorflow/` — Keras mirror that builds paired noisy/clean batches via `tf.data`, complete with PSNR metric logging.

---

## Suggested experiments

1. Re-run the vanilla autoencoder notebook, then train the denoising version and chart the PSNR improvement on corrupted inputs.
2. Swap the Gaussian noise for salt-and-pepper or masking noise to see which corruptions are easiest to remove.
3. Transfer the trained encoder into the sparse or contractive variants to study the impact of combined objectives.