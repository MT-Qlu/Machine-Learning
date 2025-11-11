# PyTorch CNN Module

Hands-on PyTorch implementation of the vanilla CNN described in the parent README. The structure mirrors what you saw in the **Supervised Learning** projects: one notebook for guided exploration, plus a reusable `src/` package for production-style work.

---

## 1. Study the notebook first

- `notebooks/cnn_pytorch.ipynb` walks through the complete pipeline:
	1. add the `src/` package to `sys.path`;
	2. train the model with automatic device selection (`mps` > `cuda` > `cpu`);
	3. inspect the metrics dictionary; and
	4. run inference on a held-out Fashion-MNIST item.
- Treat it as a lab manual—pause after each cell to inspect tensor shapes and outputs.

---

## 2. Understand the modular layout

| File | Responsibility |
| ---- | -------------- |
| `config.py` | hyperparameters, directories, and device detection |
| `data.py` | downloads Fashion-MNIST, builds training/validation dataloaders |
| `model.py` | defines `FashionMNISTCNN` (`nn.Module`) |
| `engine.py` | training + evaluation loops (same signatures as other modules) |
| `train.py` | command-line friendly entry point; saves weights & metrics |
| `inference.py` | loads checkpoints and produces predictions |
| `utils.py` | accuracy/time helpers |

Everything is type-annotated so your IDE will offer completions similar to the rest of the repository.

---

## 3. Run it yourself

```bash
python -m pip install torch torchvision
python "Deep Learning/Neural Networks/Convolutional Neural Networks/pytorch/src/train.py"
```

- Artefacts land in `artifacts/pytorch_cnn/` (weights + metrics JSON).
- Modify `CONFIG` to adjust epochs, learning rate, or dataset path—no code changes needed elsewhere.

---

## 4. Suggested exercises

1. Change the number of filters in `model.py` and re-run training; note how accuracy responds.
2. Plug the dataloaders into the `fastapi_app` project to expose predictions via REST.
3. Replace Fashion-MNIST with your own dataset (just ensure it matches the expected shape or update the model accordingly).

These micro-labs reinforce the same workflow you’ll repeat for deconvolutions, GANs, and residual networks.
