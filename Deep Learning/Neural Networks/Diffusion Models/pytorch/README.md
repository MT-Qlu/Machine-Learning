# PyTorch Diffusion Model

This directory contains a faithful PyTorch implementation of a denoising diffusion probabilistic model (DDPM). The code is designed as a learning scaffold: every mathematical step from the original paper is mirrored by a compact PyTorch function, making it straightforward to adapt the baseline to larger datasets, alternative schedules, or faster samplers.

---

## Learning outcomes

By working through the code and notebook you will:

- Derive the forward diffusion process \(q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\) and express it as vectorised PyTorch operations.
- Implement the simplified noise-prediction objective that falls out of the DDPM variational lower bound and understand why it stabilises optimisation.
- Master a modular project layout with configuration, data, model, engine, training, inference, and utility layers that promote reproducible experiments.
- Build intuition for how hyperparameters (beta schedule, sampling steps, UNet depth, optimiser settings) influence visual quality and convergence speed.

---

## Theory-to-code map

| Concept | Mathematical expression | Implementation |
| ------- | ----------------------- | -------------- |
| Beta schedule | Linear \(\beta_t\) and cumulative products \(\bar{\alpha}_t\) | `utils.py` (`make_beta_schedule`, `compute_alphas`) |
| Forward sampling | \(\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon\) | `engine.py` (`q_sample`) |
| Loss objective | \(\mathbb{E}_{t,\mathbf{x}_0,\epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t,t)\|^2]\) | `engine.py` (`p_losses`) |
| Reverse diffusion | Iterative ancestral updates with predicted noise | `engine.py` (`sample`) |
| Score network | Multi-scale UNet with sinusoidal timestep embeddings | `model.py` |
| Experiment control | Device selection, artifact paths, hyperparameters | `config.py` |

Cross-reference these modules with Ho et al. (2020) to reinforce the connection between the derivation and its implementation.

---

## Directory tour

| Component | Description |
| --------- | ----------- |
| `config.py` | Dataclass capturing all hyperparameters, artifact locations, and automatic CUDA/MPS/CPU device detection. |
| `data.py` | Downloads Fashion-MNIST, normalises it to \([-1,1]\), and yields seeded `DataLoader` instances. |
| `model.py` | Defines a lightweight UNet with residual blocks, group normalisation, and sinusoidal timestep embeddings. |
| `engine.py` | Encapsulates diffusion mathematics: schedule creation, forward sampling, loss computation, and the ancestral sampler. |
| `train.py` | Orchestrates optimisation (Adam), metric logging, checkpointing, and sample grid generation. |
| `inference.py` | Reloads checkpoints, initialises the engine, and offers a `generate_samples` helper for visualisation or downstream tasks. |
| `utils.py` | Seeding utilities, schedule helpers, metrics JSON writer, and PNG grid exporter. |
| `notebooks/diffusion_pytorch.ipynb` | Guided lab covering setup, training, reflection, sampling, and future experiment prompts. |

---

## Running the baseline

```bash
python -m pip install torch torchvision matplotlib tqdm
python "Deep Learning/Neural Networks/Diffusion Models/pytorch/src/train.py"
```

Artifacts stored under `artifacts/pytorch_diffusion/`:

- `ddpm_fashion_mnist.pt` — Model weights.
- `ddpm_samples.png` — 4x4 sample grid saved at the end of training.
- `metrics.json` — Epoch-level loss history for plotting.

To regenerate samples later:

```bash
python - <<'PY'
from Diffusion Models.pytorch.src.inference import generate_samples, CONFIG

generate_samples(CONFIG, num_images=36, output_path=CONFIG.artifact_dir / "samples_36.png")
PY
```

---

## Notebook workflow

The notebook mirrors the CLI run while adding context:

1. **Setup** – Adds `src` to the Python path, loads `CONFIG`, and prints key hyperparameters.
2. **Training** – Executes `train(CONFIG)` with progress bars, returning loss curves for immediate inspection.
3. **Analysis prompts** – Encourages plotting metrics, comparing multiple runs, and recording observations.
4. **Sampling** – Calls `generate_samples` to create a notebook-specific grid and stores it under `artifacts/`.
5. **Next experiments** – Provides follow-up ideas (schedule tweaks, network depth changes, logging improvements).

Treat the notebook as a lab journal—duplicate it per experiment and track configuration changes alongside results.

---

## Experiment playbook

1. **Schedule sweeps** – Implement cosine, quadratic, or learnable schedules in `utils.py` and compare convergence.
2. **Model capacity** – Increase base channels or insert attention layers; monitor GPU memory and sample sharpness.
3. **Classifier-free guidance** – Modify the UNet to output conditional/unconditional predictions and blend them during sampling.
4. **Alternative samplers** – Prototype DDIM or DPM-Solver steps inside `engine.py` to accelerate inference.
5. **Quantitative metrics** – Integrate FID or Inception Score to complement qualitative grids.
6. **Dataset transfer** – Swap Fashion-MNIST for EMNIST or another grayscale dataset to study generalisation.

Document objective, configuration, outcome, and interpretation for every run to build a reproducible experiment log.

---

## Troubleshooting

- **Divergent loss**: Lower the learning rate, clamp beta schedule extremes, or clip gradients.
- **Blurred samples**: Increase sampling steps or explore cosine schedules that maintain more variance early in the chain.
- **Slow iteration**: Enable AMP (`torch.cuda.amp.autocast`) or reduce batch size while prototyping.
- **Unexpected artefacts**: Verify normalisation to \([-1,1]\) and ensure checkpoints correspond to the current architecture.

---

## References

- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021.

Use these as launch points for more advanced techniques once you are comfortable with the baseline.
