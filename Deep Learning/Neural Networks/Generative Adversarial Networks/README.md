# Generative Adversarial Networks

This track dives deep into generative adversarial networks (GANs) using Fashion-MNIST as a teaching canvas. You will explore the historical context, core theory, and engineering practices required to train stable GANs in both PyTorch and TensorFlow, all while maintaining a mirrored directory structure for easy cross-referencing.

---

## Historical background

GANs, introduced by Goodfellow et al. (2014), pit two neural networks against each other in a minimax game: a generator synthesises data to fool a discriminator, while the discriminator learns to distinguish real samples from fakes. This adversarial setup has powered breakthroughs in image synthesis, domain translation, super-resolution, and more. Despite their power, GANs are notoriously finicky to train; careful architecture design, optimiser choice, and regularisation are essential. This repository crystallises best practices into accessible PyTorch and TensorFlow implementations inspired by DCGAN.

---

## Learning goals

By completing both framework tracks you will:

- Understand the adversarial objective \(\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]\) and its non-saturating variant.
- Build practical intuition for mode collapse, vanishing gradients, discriminator overpowering, and other stability challenges.
- Develop a reproducible experiment workflow that captures metrics, checkpoints, and generated grids, enabling side-by-side comparisons.
- Translate architectural heuristics (stride-2 convolutions, batch norm, LeakyReLU, latent dimensionality) into code and experiment with their effects.

---

## Directory structure

- `pytorch/` — PyTorch implementation with modular source files, training/inference utilities, and a guided notebook.
- `tensorflow/` — Keras/TensorFlow counterpart mirroring the same abstractions for device-strategy experimentation.

Each subdirectory contains:

- `src/` — Config, data pipeline, generator, discriminator, training engine, inference helpers, and utilities.
- `notebooks/` — Interactive labs that run training end-to-end and prompt analysis.
- `README.md` — Framework-specific documentation with detailed guidance, troubleshooting tips, and extension ideas.

---

## Suggested learning path

1. **Foundation** — Read the PyTorch README to understand the baseline architecture and optimisation schedule.
2. **Hands-on run** — Execute the PyTorch notebook or CLI script, inspect the loss curves, and examine the generated grids.
3. **Cross-framework transfer** — Repeat the experiment in TensorFlow, noting any behavioural differences stemming from optimiser implementations or default initialisers.
4. **Iterative refinement** — Adjust latent dimensionality, optimiser betas, label smoothing, or dropout and record the impact on stability and sample diversity.
5. **Comparative analysis** — Contrast GAN results with the diffusion models in this repository to appreciate the strengths and weaknesses of each generative paradigm.

---

## Experiment ideas

- **Conditional GANs**: Extend the data pipeline to include labels and augment both networks with embedding-based conditioning.
- **Regularisation techniques**: Implement gradient penalty, spectral normalisation, or discriminator dropout to combat overfitting and improve stability.
- **Evaluation metrics**: Add Fréchet Inception Distance (FID), Inception Score, or precision/recall measures to quantify progress.
- **Architecture variations**: Experiment with deeper generators, progressive-growing strategies, or attention mechanisms.
- **Curriculum training**: Start with lower-resolution images, gradually increase complexity, and observe training stability.

Document each experiment’s configuration, outcomes, and lessons learned—ideally within the notebooks or a dedicated research log.

---

## Further reading

- Goodfellow et al., "Generative Adversarial Networks", NeurIPS 2014.
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs", ICLR 2016.
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
- Miyato et al., "Spectral Normalization for Generative Adversarial Networks", ICLR 2018.

These works provide context and advanced techniques that you can integrate into the scaffolding provided here.
