# TensorFlow Variational Autoencoder

Keras implementation of the Fashion-MNIST VAE with KL annealing support via the config weight.

---

## Theory recap

VAEs optimise a reconstruction term plus a KL regulariser:

$$
\mathcal{L} = \mathbb{E}_{q_\theta(\mathbf{z}\mid\mathbf{x})}[\|\mathbf{x} - g_\phi(\mathbf{z})\|_2^2] + \text{KL}(q_\theta(\mathbf{z}\mid\mathbf{x})\;\|\;p(\mathbf{z}))
$$

Sampling uses $\mathbf{z} = \mu + \sigma \odot \epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$.

---

## Learning goals

- Grasp how the KL divergence constrains the latent distribution and how annealing affects training stability.
- Monitor reconstruction vs KL metrics to diagnose underfitting or posterior collapse.
- Experiment with latent dimensionality and sampling strategies to explore generative capacity.

---

## Implementation highlights

- Custom `train_step` returns reconstruction and KL losses separately for detailed logging.
- Config supports KL weight schedules, enabling annealing experiments without code changes.
- Inference utilities include sampling helpers that plug directly into notebooks for visualisation.

---

## 1. Notebook tour

- `notebooks/variational_autoencoder_tensorflow.ipynb` mirrors the configure → train → reconstruct flow and samples from the learned latent space.
- The notebook highlights the KL divergence metric and how it competes with reconstruction.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, latent size, KL weight, device visibility |
| `data.py` | Fashion-MNIST loader returning `(input, target)` pairs |
| `model.py` | Sub-classed VAE with custom training step |
| `train.py` | Compiles, trains, checkpoints, and logs metrics |
| `inference.py` | Reconstruction and sampling helpers |
| `utils.py` | KL helper, PSNR metric, history serialisation |

---

## 3. Run it

```bash
python -m pip install tensorflow matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Variational Autoencoder/tensorflow/src/train.py"
```

Artefacts: `artifacts/tensorflow_variational_ae/variational_autoencoder.weights.h5` and `metrics.json`.

---

## 4. Practice prompts

1. Lower `kl_weight` to encourage sharper reconstructions, then anneal it back.
2. Increase `latent_dim` and visualise interpolations between random samples.
3. Freeze the encoder and fine-tune only the decoder for a handful of epochs.