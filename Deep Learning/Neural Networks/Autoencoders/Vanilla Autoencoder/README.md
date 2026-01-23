# Vanilla Autoencoder

The introductory autoencoder module pairs theory with practical PyTorch and TensorFlow implementations for Fashion-MNIST. It sets the stage for the denoising, sparse, contractive, and variational variants found alongside this directory.

---

## Theory primer

The vanilla autoencoder learns a compressed representation by minimising reconstruction error:

$$
\mathbf{z} = f_\theta(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\phi(\mathbf{z})
$$

$$
\mathcal{L}_{\text{rec}}(\theta,\phi) = \mathbb{E}_{\mathbf{x} \sim p_{data}}\big[\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2\big]
$$

With a bottleneck ($\dim(\mathbf{z}) \ll \dim(\mathbf{x})$), the model is forced to capture the most salient structure of the data. Increasing latent size generally improves reconstruction but can reduce the usefulness of the learned features (the model may simply learn an identity mapping).

### Practical intuition

- **Undercomplete AE**: smaller latent space encourages compact, structured representations.
- **Overcomplete AE**: larger latent space requires additional regularisation (sparsity, noise, contraction) to avoid trivial solutions.
- **Loss choice**: MSE emphasises pixel-wise fidelity; MAE can produce sharper reconstructions.

---

## Learning goals

- Understand the reconstruction objective that underpins the broader autoencoder family.
- Explore how latent dimensionality balances compression against reconstruction fidelity.
- Establish a reference training/evaluation workflow used by more advanced variants.

---

## What's inside?

- `pytorch/` — modular training package + notebook leveraging Torch's automatic device selection (MPS → CUDA → CPU).
- `tensorflow/` — parallel Keras stack with a custom PSNR metric and scripted checkpoints.
- Each framework folder contains a notebook mirroring the same three-step routine: configure → train → reconstruct.

---

## Suggested progression

1. Run either notebook end-to-end to familiarise yourself with the modular layout.
2. Experiment with latent sizes or hidden layer widths to see how reconstructions degrade or improve.
3. Move on to `Denoising Autoencoder/` to extend the baseline with input corruption.
