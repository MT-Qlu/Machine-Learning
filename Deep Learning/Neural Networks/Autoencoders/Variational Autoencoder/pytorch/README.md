````markdown
# PyTorch Variational Autoencoder

Latent variable model that learns a Gaussian posterior for Fashion-MNIST.

---

## Learning goals

- Understand how the KL divergence term regularises the latent space and enforces approximate posterior structure.
- Visualise latent traversals and reconstructions to see the balance between reconstruction loss and KL weight.
- Explore how changing `latent_dim` or `kl_weight` affects sample diversity and reconstruction fidelity.

---

## Implementation highlights

- Encoder exposes `mu` and `logvar` so you can probe distributions directly in notebooks.
- Training loop logs KL and reconstruction losses separately, making the trade-off easy to monitor.
- Inference module ships sampling helpers that reuse the trained decoder for quick generation experiments.

---

## 1. Notebook tour

- `notebooks/variational_autoencoder_pytorch.ipynb` demonstrates training, inspects KL divergence, and samples novel digits.
- The notebook mirrors the same modular usage pattern as the other PyTorch autoencoders.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, latent size, and KL weight |
| `data.py` | Fashion-MNIST loaders |
| `model.py` | VAE with `mu/logvar` heads and reparameterisation trick |
| `engine.py` | Training loop combining reconstruction and KL losses |
| `train.py` | Entry point returning final metrics |
| `inference.py` | Utilities for reconstruction and sampling |
| `utils.py` | Seeding, PSNR, KL helper, metrics persistence |

---

## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Variational Autoencoder/pytorch/src/train.py"
```

Artefacts: `artifacts/pytorch_variational_ae/variational_autoencoder.pt` and `metrics.json`.

---

## 4. Practice prompts

1. Increase `latent_dim` to 64 and compare generated samples.
2. Reduce `kl_weight` to ease the KL pressure and observe the trade-off.
3. Use `inference.sample` to create a grid of synthetic images.

````