# Autoencoders Roadmap

Comprehensive suite of Fashion-MNIST autoencoders implemented in both PyTorch and TensorFlow. Each sub-directory is self-contained with modular source code, documentation, and notebooks that follow a common workflow: configure → train → reconstruct → experiment.

---

## Core theory (shared across variants)

An autoencoder is a neural network that learns a compressed representation by optimising a reconstruction objective. It consists of an **encoder** $f_\theta$ and **decoder** $g_\phi$:

$$
\mathbf{z} = f_\theta(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\phi(\mathbf{z})
$$

The baseline objective minimises reconstruction error (often mean squared error for images):

$$
\mathcal{L}_{\text{rec}}(\theta,\phi) = \mathbb{E}_{\mathbf{x} \sim p_{data}}\big[\|\mathbf{x} - g_\phi(f_\theta(\mathbf{x}))\|_2^2\big]
$$

Autoencoders learn **useful latent structure** by forcing information through a bottleneck ($\dim(\mathbf{z}) \ll \dim(\mathbf{x})$) or by adding regularisers (noise, sparsity, contraction, KL divergence) that shape the representation.

### Variant-specific objectives (at a glance)

- **Denoising**: reconstruct clean inputs from corrupted versions.
  $$
  \mathcal{L} = \mathbb{E}_{\mathbf{x},\tilde{\mathbf{x}}}\big[\|\mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}}))\|_2^2\big]
  $$
- **Sparse**: encourage low average activation per latent unit.
  $$
  \mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \sum_j \text{KL}(\rho \|\, \hat{\rho}_j)
  $$
- **Contractive**: penalise sensitivity of the encoder to input perturbations.
  $$
  \mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \|\nabla_\mathbf{x} f_\theta(\mathbf{x})\|_F^2
  $$
- **Variational**: learn a probabilistic latent distribution.
  $$
  \mathcal{L} = \mathbb{E}[\mathcal{L}_{\text{rec}}] + \text{KL}(q_\theta(\mathbf{z}\mid\mathbf{x})\;\|\;p(\mathbf{z}))
  $$

| Variant | Motivation | Notes |
| ------- | ---------- | ----- |
| `Vanilla Autoencoder/` | Baseline reconstruction objective | Complete PyTorch + TensorFlow stacks |
| `Denoising Autoencoder/` | Robustness to Gaussian corruption | Noise injected in the dataloaders / `tf.data` pipeline |
| `Sparse Autoencoder/` | Encourage sparse latent activations | KL sparsity penalty with monitoring metrics |
| `Contractive Autoencoder/` | Penalise encoder sensitivity | Analytic Jacobian penalty for robustness |
| `Variational Autoencoder/` | Probabilistic latent space | Sampling helpers + KL tracking |

### Learning goals

- Build intuition for how different objectives (noise, sparsity, contraction, KL) shape latent representations.
- Practise running parallel experiments in PyTorch and TensorFlow using the same project structure.
- Develop a reusable workflow for training, evaluating, and visualising autoencoder behaviour.

### Implementation highlights

- Every module exposes consistent `config/data/model/train/inference/utils` packages to streamline experimentation.
- Notebooks complement the code with hands-on walkthroughs and recommended probes.
- Training scripts emit metrics JSONs plus checkpoints under `artifacts/`, simplifying cross-run comparisons.

### How to use this section

1. Start with the vanilla module to familiarise yourself with the shared package layout.
2. Progress through denoising, sparse, contractive, and variational variants to explore increasingly advanced objectives.
3. Each module offers both PyTorch (`pytorch/`) and TensorFlow (`tensorflow/`) implementations, mirroring docs and notebooks for easy cross-framework comparisons.
4. Artefacts are saved under `artifacts/<framework>_<variant>/`, making it simple to compare metrics or reconstructions across experiments.

Looking for next steps? Extend these templates to convolutional autoencoders, add attention blocks, or integrate the training utilities into your own datasets by editing the config/data modules.