# TensorFlow Deconvolutional Autoencoder

TensorFlow/Keras implementation of the convolutional autoencoder described in the parent README. Use it alongside the PyTorch version to compare tooling and APIs.

---

## 1. Notebook walkthrough

- `notebooks/deconv_tensorflow.ipynb` mirrors the PyTorch notebook:
  - mount `../src` on `sys.path`;
  - call `train.train()` to fit the autoencoder (with MPS/GPU/CPU selection handled in `config.py`);
  - reconstruct a Fashion-MNIST sample and inspect the output.
- The notebook highlights how `tf.data` streams batches and how `.predict()` generates reconstructions.

---

## 2. Package structure

| File | Role |
| ---- | ---- |
| `config.py` | directories, hyperparameters, and device visibility |
| `data.py` | builds `tf.data.Dataset` pipelines returning `(image, image)` pairs |
| `model.py` | defines the Keras autoencoder with `Conv2DTranspose` layers |
| `utils.py` | compilation helper + PSNR metric + history persistence |
| `train.py` | scripted training with ModelCheckpoint |
| `inference.py` | loads the saved `.h5` model and reconstructs numpy arrays |

---

## 3. Run the trainer

```bash
python -m pip install tensorflow-macos tensorflow-metal
python "Deep Learning/Neural Networks/Deconvolutional neural networks/tensorflow/src/train.py"
```

Artefacts (weights + metrics) land in `artifacts/tensorflow_deconv/`.

---

## 4. Suggested experiments

1. Add dropout or layer-normalisation layers to the decoder and observe reconstruction quality.
2. Introduce light data augmentation (random rotations) in `data.py` to test robustness.
3. Export the trained model to TensorFlow Lite and benchmark inference speed on mobile hardware.

Use the findings to inform future decoder architectures such as U-Net or GAN generators.
