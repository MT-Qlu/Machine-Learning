# TensorFlow GAN

This directory recreates the DCGAN-inspired Fashion-MNIST baseline in TensorFlow/Keras. It serves as a companion to the PyTorch implementation, highlighting how to express adversarial training loops with `tf.GradientTape`, leverage distribution strategies, and manage experiment artefacts in the TensorFlow ecosystem.

---

## Learning objectives

You will learn to:

- Implement generator and discriminator networks with Keras layers, mirroring DCGAN architectural heuristics.
- Write custom training steps using `GradientTape`, ensuring consistent optimisation behaviour with the PyTorch engine.
- Manage experiment artefacts (checkpoints, metrics, sample grids) in a reproducible fashion.
- Explore TensorFlow-specific performance tools such as mixed precision and distributed training strategies.

---

## Source tour

| File | Description |
| ---- | ----------- |
| `config.py` | Hyperparameters, artifact paths, and a convenience accessor for the current distribution strategy. |
| `data.py` | Loads Fashion-MNIST, scales to \([-1,1]\), and returns batched `tf.data.Dataset` pipelines with shuffling and prefetching. |
| `model.py` | Builds generator and discriminator Keras models with transpose convolutions, batch norm, and LeakyReLU activations. |
| `engine.py` | Defines discriminator and generator steps using binary cross-entropy losses wrapped in `GradientTape`. |
| `train.py` | Coordinates the training loop, logs losses, saves checkpoints, and produces sample grids. |
| `inference.py` | Reloads generator weights and exposes a `generate_samples` helper. |
| `utils.py` | Provides seeding, metrics persistence, and matplotlib-based image grid utilities. |
| `notebooks/gan_tensorflow.ipynb` | Guided notebook that runs training, sampling, and reflection prompts. |

---

## Running the baseline

```bash
python -m pip install tensorflow tqdm matplotlib
python "Deep Learning/Neural Networks/Generative Adversarial Networks/tensorflow/src/train.py"
```

Artefacts are saved to `artifacts/tensorflow_gan/`:

- `generator.*` — Saved generator weights.
- `discriminator.*` — Saved discriminator weights.
- `gan_samples.png` — Sample grid created after training.
- `metrics.json` — Epoch-level loss history for later analysis.

Generate new samples at any time with:

```bash
python - <<'PY'
from Generative Adversarial Networks.tensorflow.src.inference import generate_samples, CONFIG

images = generate_samples(CONFIG, num_images=36, output_path=CONFIG.artifact_dir / "samples_36.png")
print(images.shape)
PY
```

---

## Notebook agenda

`notebooks/gan_tensorflow.ipynb` mirrors the CLI while providing guidance:

1. **Setup** – Imports TensorFlow modules, adds `src/` to `sys.path`, and inspects the active strategy.
2. **Training** – Runs `train(CONFIG)` and captures the resulting loss dictionary for charting.
3. **Reflection** – Encourages analysis of loss oscillations, stability, and convergence patterns.
4. **Sampling** – Demonstrates `generate_samples` to create a notebook-specific sample grid.
5. **Next experiments** – Suggests tuning optimisers, enabling mixed precision, or adding regularisation.

---

## Experiment playbook

1. **Mixed precision** – Enable `tf.keras.mixed_precision.set_global_policy("mixed_float16")` to accelerate GPU/TPU workloads.
2. **Optimiser sweeps** – Compare Adam, RMSProp, or AdamW; adjust betas or weight decay to improve stability.
3. **Regularisation** – Implement gradient penalty, spectral normalisation, or instance noise to reduce mode collapse.
4. **Conditional GAN** – Incorporate label embeddings in both networks to generate class-conditional samples.
5. **Evaluation metrics** – Add FID or Inception Score computations using TensorFlow Hub models.
6. **Distributed training** – Wrap the training loop in `tf.distribute.MirroredStrategy()` for multi-GPU setups.

Record configuration changes, seeds, and outcomes in the notebook to maintain a reproducible log.

---

## Troubleshooting

- **Loss spikes**: Lower the discriminator learning rate, add label smoothing, or update the generator multiple times per discriminator step.
- **Mode collapse**: Inject instance noise, apply adaptive discriminator augmentation, or experiment with gradient penalty.
- **Training stalls**: Confirm gradient flow by inspecting `GradientTape` outputs and avoid accidentally reusing tapes.
- **Checkerboard artefacts**: Replace transpose convolutions with upsample + convolution blocks.

---

## References

- Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs", ICLR 2016.
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.

Consult these works as you iterate towards more advanced architectures or training schemes.
