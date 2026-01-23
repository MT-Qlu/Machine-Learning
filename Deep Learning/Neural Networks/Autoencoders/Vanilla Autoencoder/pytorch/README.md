# PyTorch Vanilla Autoencoder

Train a fully-connected autoencoder on Fashion-MNIST using a minimal modular package.


## Learning goals

- Understand how reconstruction loss trains the encoder/decoder pair.
- See how latent size impacts compression quality and PSNR.
- Use the modular pipeline to reproduce results quickly.


## Implementation highlights

- Clean separation of config, data, model, engine, and inference modules.
- Automatic device selection (MPS → CUDA → CPU).
- Metrics JSON + checkpoints for reproducible comparisons.


## 1. Notebook tour

- `notebooks/vanilla_autoencoder_pytorch.ipynb` walks through configure → train → reconstruct.
- The notebook highlights PSNR curves and qualitative reconstructions.

## Beginner example

Imagine compressing a drawing into a short code and then reconstructing it. The autoencoder learns the best code so the reconstruction looks almost the same as the original.


## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, device detection, and artifact paths |
| `data.py` | Fashion-MNIST loaders with standard normalisation |
| `model.py` | Fully-connected encoder/decoder with configurable latent size |
| `engine.py` | Training + evaluation loops returning MSE and PSNR |
| `train.py` | High-level entry point for CLI / notebook usage |
| `inference.py` | Lightweight helpers for checkpoint loading + reconstruction |
| `utils.py` | Seeding, PSNR calculation, and metric serialisation |


## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Vanilla Autoencoder/pytorch/src/train.py"
```

Weights and metrics land in `artifacts/pytorch_vanilla_ae/` (`vanilla_autoencoder.pt`, `metrics.json`).


## 4. Practice prompts

1. Swap the hidden dimensions or latent size to explore reconstruction quality vs dimensionality.
2. Add dropout layers to the encoder and observe how PSNR evolves across epochs.
3. Replace the MLP with a convolutional autoencoder by editing `model.py` and compare qualitative outputs.


## Theory recap

The vanilla autoencoder minimises reconstruction error by learning an encoder $f_\theta$ and decoder $g_\phi$:

$$
\mathbf{z} = f_\theta(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\phi(\mathbf{z})
$$

$$
\mathcal{L}_{\text{rec}} = \mathbb{E}_{\mathbf{x}}\big[\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2\big]
$$

With a bottleneck, the model must compress signal into $\mathbf{z}$, yielding compact features rather than an identity map.
