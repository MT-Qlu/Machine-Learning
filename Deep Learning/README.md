# Deep Learning Roadmap

> Scope: core deep-learning families outside of the dedicated Computer Vision and Natural Language Processing repositories.

---

## Foundational Tracks

- **PyTorch Basics** — tensor operations, autograd, training loops, custom datasets, and deployment patterns.
- **TensorFlow Basics** — Keras workflow, data input pipelines, and classification tutorials.
- **Neural Networks** — architecture-specific modules (see directory map below).
- **Neural Architecture Search** — automated model discovery, NAS algorithms, and search-space experiments.

## Architecture Registry (`Neural Networks/`)

- **MultiLayer Perceptrons (Feed Forward)** — fully connected baselines for tabular tasks.
- **Convolutional Neural Networks** — non-CV applications (e.g., time series) and cross-repo references.
- **Deconvolutional Neural Networks** — decoder blocks, generative upsampling.
- **Recurrent Neural Networks**
  - Long Short-Term Memory
  - Gated Recurrent Unit
- **Residual Networks** — skip-connection patterns beyond image tasks.
- **Graph Neural Networks** — node/edge/graph-level learning.
- **Generative Adversarial Networks** — GAN variants and training tricks.
- **Boltzmann & Hopfield Machines** — energy-based models.
- **Autoencoders**
  - Vanilla Autoencoder
  - Denoising Autoencoder
  - Sparse Autoencoder
  - Contractive Autoencoder
  - Variational Autoencoder
- **Normalizing Flows** — invertible density models.
- **Diffusion Models** — score-based and denoising diffusion frameworks.
- **Transformers** — attention mechanisms, sequence-to-sequence, and general-purpose encoder/decoder stacks.
- **Self-Supervised Learning** — contrastive, masked modeling, BYOL/SimCLR families.
- **Meta Learning** — model-agnostic meta-learning, few-shot adapters.
- **Continual Learning** — rehearsal, regularization, dynamic architectures.
- **Reinforcement Learning** — deep RL algorithms (policy/value-based, actor-critic).

## How to Extend

1. **Create Submodule** — add `data/`, `notebooks/`, `src/`, and `artifacts/` folders within the target architecture.
2. **Document Usage** — include a README with theory, training steps, and serving guidance.
3. **Integrate Metrics** — reuse `Evaluation/` playbooks and standardized metric helpers.
4. **Hook into Serving** — expose inference endpoints via `fastapi_app` where applicable.

## Next Steps

- Populate each architecture with minimal PyTorch/TensorFlow baselines.
- Mirror production-grade pipelines (training, evaluation, inference) following supervised-learning modules.
- Cross-link to specialized repositories (CV, NLP) for domain-specific deep learning implementations.
