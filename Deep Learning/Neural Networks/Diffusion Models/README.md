# Diffusion Models

This track provides an end-to-end laboratory for denoising diffusion probabilistic models (DDPMs), combining historical background, mathematical intuition, and practical engineering guidance. You will move from fundamental stochastic processes to hands-on experimentation in both PyTorch and TensorFlow, all while working with a common project structure that makes cross-framework comparisons straightforward.

---

## Why diffusion models?

Diffusion-based generators have rapidly become the state of the art in high-fidelity image, audio, and video synthesis. They combine the interpretability of probabilistic graphical models with the flexibility of deep neural networks, offering:

- **Stable training** by optimising a variational bound that decomposes into simple denoising objectives.
- **High sample quality** that rivals or surpasses GANs on perceptual benchmarks when paired with suitable samplers.
- **Strong theoretical grounding** through connections to score matching, stochastic differential equations, and nonequilibrium thermodynamics.

This learning path distils those ideas into approachable Fashion-MNIST experiments, equipping you to scale up to more complex domains.

---

## Conceptual overview

1. **Forward diffusion (noising)**: Starting from a clean sample \( \mathbf{x}_0 \), repeatedly add Gaussian noise according to a predefined schedule \( \{\beta_t\}_{t=1}^T \) until the signal becomes nearly isotropic noise. We model the process as
	\[
	q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I}).
	\]

2. **Reverse diffusion (denoising)**: Train a neural network to approximate the reverse transitions \( p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \). Because these transitions are intractable analytically, the model learns to predict the noise injected at each step, effectively estimating the score function.

3. **Training objective**: Minimise the mean squared error between the true noise and the model-predicted noise. This objective emerges from maximising an evidence lower bound (ELBO) on log-likelihood and yields a simple per-timestep regression target.

4. **Sampling**: Starting from pure noise, iteratively apply the learned reverse transitions to recover a clean sample \( \mathbf{x}_0 \). Faster samplers (DDIM, PLMS, DPM-Solver) can accelerate this procedure; the baseline provided here implements the classical ancestral sampler for clarity.

---

## Repository layout

- `pytorch/` — PyTorch implementation featuring a lightweight UNet, modular engine, and reproducible training script.
- `tensorflow/` — TensorFlow/Keras counterpart mirroring the same abstractions for device strategy experiments.
- Shared conventions (`src/config.py`, `src/data.py`, `src/model.py`, `src/engine.py`, `src/train.py`, `src/inference.py`, `src/utils.py`, `notebooks/`) make it easy to transfer ideas across frameworks.

---

## Learning goals

By completing both tracks you will:

- Internalise the mechanics of forward and reverse diffusion processes, including schedule design and score estimation.
- Translate the mathematical training objective into concrete code, gaining intuition for how loss magnitude aligns with visual quality.
- Develop a modular experimentation workflow that logs metrics, checkpoints, and sample grids for each run.
- Compare framework ergonomics, device support, and numerical behaviour when implementing identical algorithms.

---

## What you will build

1. **Dataset pipeline** that downloads Fashion-MNIST, scales images to \([-1, 1]\), and yields batched tensors ready for GPU/TPU execution.
2. **UNet backbone** with sinusoidal timestep embeddings to capture multi-scale context during denoising.
3. **Diffusion engine** encapsulating beta schedules, posterior parameterisation, sampling loops, and loss computation.
4. **Training CLI** that tracks progress with TQDM, saves metrics to JSON, and persists weights for later inference.
5. **Inference helper** to reload checkpoints and generate visual grids for qualitative evaluation.
6. **Interactive notebooks** guiding you through each phase with reflection prompts and experimentation ideas.

---

## Suggested progression

1. **Deep dive the theory**: Read the PyTorch README to understand the implementation details, then inspect `engine.py` to see how the equations map to code.
2. **Execute PyTorch pipeline**: Run the notebook or `train.py`, inspect the metrics, and iterate on hyperparameters (epochs, learning rate, beta schedule).
3. **Mirror in TensorFlow**: Repeat the same experiment using the TensorFlow stack, noting differences in optimiser configuration, distribution strategies, or numerical stability.
4. **Experiment**: Introduce alternative noise schedules, try classifier-free guidance, or swap Fashion-MNIST for another grayscale dataset.
5. **Report results**: Save sample grids from both frameworks, plot loss curves, and document findings using the provided prompts.

---

## Extending the project

- **Alternative samplers**: Implement DDIM or DPM-Solver sampling steps to reduce inference time while monitoring quality.
- **Conditioning**: Add label information by concatenating embeddings to the UNet, enabling class-conditional generation.
- **Evaluation metrics**: Integrate Fréchet Inception Distance (FID) or Inception Score for quantitative comparison across runs.
- **Larger datasets**: Scale to CIFAR-10 or CelebA by adjusting image size, network capacity, and training schedule.
- **Continuous-time diffusion**: Explore stochastic differential equation (SDE) formulations and score-based generative modelling for advanced topics.

---

## Further reading

- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022)

Use these papers to deepen your theoretical understanding before adapting the code to larger-scale projects.
