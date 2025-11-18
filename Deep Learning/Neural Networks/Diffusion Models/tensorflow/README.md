# TensorFlow Diffusion Model

This directory mirrors the PyTorch DDPM implementation using TensorFlow and Keras. It demonstrates how to translate diffusion modelling concepts into the TensorFlow ecosystem, take advantage of distribution strategies, and prepare the groundwork for scaling to TPUs or mixed-precision workflows.

---

## Learning objectives

By completing the TensorFlow track you will:

- Recreate the forward and reverse diffusion processes using native TensorFlow operations and automatic differentiation.
- Implement a custom training loop with `tf.GradientTape`, ensuring parity with the PyTorch baseline.
- Understand how to leverage `tf.data` pipelines, distribution strategies, and mixed precision to accelerate experimentation.
- Compare numerical behaviour, logging ergonomics, and device support between TensorFlow and PyTorch implementations of the same algorithm.

---

## Component overview

| Component | Description |
| --------- | ----------- |
| `config.py` | Dataclass capturing hyperparameters, artifact paths, and an accessor for the active `tf.distribute.Strategy`. |
| `data.py` | Downloads Fashion-MNIST, scales images to \([-1,1]\), and returns shuffled, batched `tf.data.Dataset` pipelines. |
| `model.py` | Builds a Keras UNet with transpose convolutions, residual blocks, and sinusoidal timestep embeddings. |
| `engine.py` | Encapsulates noise schedule creation, forward sampling, loss computation, and ancestral sampling in pure TensorFlow. |
| `train.py` | Orchestrates the training loop with `GradientTape`, Adam optimisers, metric aggregation, checkpointing, and sample export. |
| `inference.py` | Reloads generator weights, constructs the diffusion engine, and produces image grids for qualitative evaluation. |
| `utils.py` | Factors out seeding, schedule utilities, metric persistence, and matplotlib-based grid rendering. |
| `notebooks/diffusion_tensorflow.ipynb` | Guided lab that walks through setup, training, sample generation, and reflection prompts. |

---

## Running the pipeline

```bash
python -m pip install tensorflow matplotlib tqdm
python "Deep Learning/Neural Networks/Diffusion Models/tensorflow/src/train.py"
```

Artefacts are stored in `artifacts/tensorflow_diffusion/`:

- `checkpoints/` — Saved generator weights.
- `ddpm_samples.png` — Sample grid generated after training.
- `metrics.json` — Epoch-level training and validation loss history.

Sampling on demand:

```bash
python - <<'PY'
from Diffusion Models.tensorflow.src.inference import generate_samples, CONFIG

generate_samples(CONFIG, num_images=36, output_path=CONFIG.artifact_dir / "samples_36.png")
PY
```

---

## Notebook agenda

The notebook follows the same structure as the CLI script but embeds discussion points:

1. **Setup** – Adds `src` to the Python path, imports TensorFlow modules, and inspects the active distribution strategy.
2. **Training** – Executes `train(CONFIG)` while capturing the returned metrics dictionary for inspection or plotting.
3. **Reflection** – Prompts you to analyse the loss curves, compare runs, and reason about convergence in the TensorFlow context.
4. **Sampling** – Demonstrates `generate_samples` and saves the resulting grid to `artifacts/` for visual comparison.
5. **Next experiments** – Suggests exploring mixed precision, alternative schedules, and TensorBoard logging.

Use TensorBoard or Weights & Biases to log notebook runs if you plan to iterate frequently.

---

## Experiment ideas

1. **Mixed precision** – Enable `tf.keras.mixed_precision.set_global_policy("mixed_float16")` to benchmark speed-ups on GPU/TPU hardware.
2. **Schedule exploration** – Add cosine or learnable schedules in `engine.py`, ensuring numerical stability by clipping extremes.
3. **Sampler variations** – Prototype DDIM sampling in TensorFlow and compare latency/quality trade-offs with the ancestral sampler.
4. **Multi-device training** – Wrap the training code with `tf.distribute.MirroredStrategy()` to scale across multiple GPUs.
5. **Quantitative metrics** – Integrate FID computation using TensorFlow Hub Inception networks for automated evaluation.
6. **Dataset portability** – Swap Fashion-MNIST for other grayscale datasets (EMNIST, MedNIST) and adjust image size accordingly.

Document configuration changes and outcomes within the notebook to build a reproducible experiment log.

---

## Troubleshooting

- **Diverging loss**: Lower the learning rate, clip gradients, or adjust the beta schedule to avoid overly aggressive noise early in training.
- **Numerical instability**: Ensure operations run in float32 even when mixed precision is enabled; cast as necessary.
- **Slow input pipeline**: Add `.cache()` and tune parallelism arguments in `tf.data` to keep devices boundless.
- **Blurry samples**: Increase sampling steps or experiment with cosine schedules to maintain higher signal-to-noise ratios mid-chain.

---

## Further reading

- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021.
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.

Once you are comfortable with the TensorFlow baseline, these resources will guide you towards more advanced architectures and sampling schemes.
