# PyTorch Deconvolutional Autoencoder

Teach yourself transposed convolutions by training a small convolutional autoencoder on Fashion-MNIST.

---

## 1. Notebook tour

- `notebooks/deconv_pytorch.ipynb` mirrors the step-by-step pattern used in other modules:
  1. add `../src` to `sys.path`;
  2. run `train.train()` to fit the autoencoder (device auto-detects `mps` → `cuda` → `cpu`);
  3. reconstruct a sample image and compare it with the original.
- Inspect intermediate outputs to verify that the decoder truly upsamples the latent representation.

---

## 2. Source layout

| File | Role |
| ---- | ---- |
| `config.py` | directories, hyperparameters, device detection |
| `data.py` | Fashion-MNIST dataset + dataloaders that return image/label pairs (labels unused) |
| `model.py` | convolutional encoder + transposed-convolution decoder |
| `engine.py` | train/eval loops that report MSE loss and PSNR |
| `train.py` | CLI-friendly training entry point that saves weights + metrics |
| `inference.py` | helper to load checkpoints and reconstruct tensors |
| `utils.py` | PSNR calculation + formatting helpers |

---

## 3. Run it

```bash
python -m pip install torch torchvision
python "Deep Learning/Neural Networks/Deconvolutional neural networks/pytorch/src/train.py"
```

Artefacts are written to `artifacts/pytorch_deconv/` (`deconv_autoencoder.pt` + `metrics.json`).

---

## 4. Practice prompts

1. Increase the latent dimensionality (add another encoder layer) and track how PSNR behaves.
2. Replace MSE with `nn.L1Loss()` and compare reconstruction sharpness.
3. Export a few reconstructed images with matplotlib to visually inspect checkerboard artifacts.

The exercises prepare you for richer decoder architectures (U-Net, GAN generators, diffusion decoders) later in the Deep Learning roadmap.
