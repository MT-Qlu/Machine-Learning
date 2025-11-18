# PyTorch GAN

The PyTorch implementation encapsulates a DCGAN-inspired architecture tailored to Fashion-MNIST. It emphasises clean abstractions for generator/discriminator construction, adversarial training steps, and experiment logging so you can iterate rapidly and understand how architectural and optimiser choices influence stability.

---

## Learning outcomes

- Internalise the non-saturating GAN objective and track how generator/discriminator losses behave during training.
- Understand the role of transposed convolutions, batch normalisation, and LeakyReLU activations in stabilising GAN optimisation.
- Build a reproducible experiment setup that saves checkpoints, metrics, and generated grids for comparison.
- Diagnose common GAN pathologies (mode collapse, gradient vanishing, discriminator overpowering) and test mitigation strategies.

---

## Theory recap

The baseline uses the non-saturating loss formulation:

\[
\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
\]

\[
\mathcal{L}_G = -\mathbb{E}_{z \sim p(z)}[\log D(G(z))]
\]

where the generator receives gradients through the discriminator’s predictions. The implementation follows the DCGAN heuristics (use of Adam with \(\beta_1 = 0.5\), strided convolutions, batch norm) that promote stable convergence.

---

## Source tour

| File | Description |
| ---- | ----------- |
| `config.py` | Hyperparameters (batch size, latent dim, learning rates, betas), device selection, and artifact paths. |
| `data.py` | Downloads Fashion-MNIST, normalises to \([-1,1]\), and returns a shuffled `DataLoader`. |
| `model.py` | Defines the generator (latent to 28x28 image) and discriminator (28x28 image to scalar logit). |
| `engine.py` | Implements discriminator and generator training steps with binary cross-entropy losses. |
| `train.py` | Runs the epoch loop, tracks losses, saves checkpoints, and emits sample grids. |
| `inference.py` | Loads generator weights and exposes a `generate_samples` helper. |
| `utils.py` | Houses seeding utilities, metric writers, and image grid savers. |
| `notebooks/gan_pytorch.ipynb` | Guided notebook that executes the workflow and prompts analysis. |

---

## Running the pipeline

```bash
python -m pip install torch torchvision tqdm matplotlib
python "Deep Learning/Neural Networks/Generative Adversarial Networks/pytorch/src/train.py"
```

Artifacts written to `artifacts/pytorch_gan/`:

- `generator.pt`
- `discriminator.pt`
- `gan_samples.png`
- `metrics.json`

To generate new samples later:

```bash
python - <<'PY'
from Generative Adversarial Networks.pytorch.src.inference import generate_samples, CONFIG

images = generate_samples(CONFIG, num_images=36, output_path=CONFIG.artifact_dir / "samples_36.png")
print(images.shape)
PY
```

---

## Notebook workflow

The notebook mirrors the CLI script but layers educational prompts:

1. **Setup** – Adds `src/` to the Python path and loads configuration.
2. **Training** – Executes `train(CONFIG)` with progress bars and returns the metrics dictionary.
3. **Reflection** – Encourages plotting loss curves, noting oscillations, and diagnosing instability.
4. **Sampling** – Demonstrates `generate_samples` to visualise results within the notebook.
5. **Next experiments** – Offers ideas for latent sweeps, regularisation, or evaluation metrics.

---

## Experiment playbook

1. **Latent dimensionality** – Modify `CONFIG.latent_dim` to study diversity vs. convergence speed.
2. **Optimiser tuning** – Adjust learning rates or betas; experiment with RMSProp or AdamW for stability.
3. **Regularisation** – Introduce label smoothing, instance noise, or gradient penalty to mitigate discriminator dominance.
4. **Conditional generation** – Append label embeddings to generator inputs and discriminator features to create a conditional GAN.
5. **Evaluation metrics** – Add FID or Inception Score computation to quantify progress beyond visual inspection.
6. **Architecture tweaks** – Swap transpose convolutions for upsampling + convolution to reduce checkerboard artefacts.

Record objectives, hyperparameters, and outcomes for each run—ideally within the notebook prompts—to build a reproducible research log.

---

## Troubleshooting tips

- **Generator collapse**: Reduce the discriminator learning rate or apply one-sided label smoothing (use 0.9 for real labels).
- **Discriminator overpowering**: Train the generator multiple steps per discriminator step or add dropout to the discriminator.
- **Slow convergence**: Increase training epochs, warm up with higher latent variance, or monitor if gradients vanish (check `.grad` norms).
- **Checkerboard artefacts**: Replace transpose convolutions with nearest-neighbour upsampling followed by convolution layers.

---

## References

- Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs", ICLR 2016.
- Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016.
- Miyato et al., "Spectral Normalization for Generative Adversarial Networks", ICLR 2018.

Use these works to inform further improvements once you master the baseline implementation.
