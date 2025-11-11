# Deconvolutional Neural Networks (Transposed Convolutions)

> **Goal**: understand how neural networks learn to upsample feature maps, a critical component in decoders, segmentation heads, and generative models.

This module mirrors the teaching style of the CNN chapter: begin with intuition, explore the maths, then experiment using PyTorch and TensorFlow implementations (notebooks + modular code). We focus on a classic convolutional autoencoder that compresses Fashion-MNIST images and reconstructs them via learned transposed convolutions.

---

## 1. Why transposed convolutions?

- **Upsampling with learnable parameters**: while nearest-neighbour or bilinear upsampling copy pixels, transposed convolutions learn how to fill in detail.
- **Decoder counterpart to CNN encoders**: autoencoders, U-Nets, and GAN generators rely on learned upsampling to recover spatial resolution.
- **Bridge to pixel-wise tasks**: segmentation, super-resolution, and image synthesis all hinge on accurate reconstruction from compressed features.

---

## 2. Intuition and maths

Think of a standard convolution as sliding a filter over an image and producing a smaller (or equal-sized) feature map. A **transposed convolution** performs the inverse operation: it takes a low-resolution feature map and produces a higher-resolution output by inserting learnable weights between pixels.

For a 1D example with stride \(s\) and kernel \(W\):

1. Insert \(s-1\) zeros between input elements (the "stride trick").
2. Convolve with the flipped kernel \(W\).
3. Crop/pad to the desired output shape.

In matrix form, a standard convolution is \(y = Kx\). The transposed convolution multiplies by \(K^\top\), redistributing each input activation into multiple output positions. The 2D case adds zero-padding between rows and columns but follows the same principle.

**Padding and stride pitfalls**

- Output size is determined by kernel, stride, and padding. Mismatched settings cause checkerboard artifacts.
- Even/odd kernel sizes affect alignment. Practice by computing the formula: `output = (input - 1) * stride - 2 * padding + kernel`.

---

## 3. Baseline architecture

We implement a shallow convolutional autoencoder:

```
Encoder
  Conv2D(1 → 32, 3×3, stride=2) → ReLU
  Conv2D(32 → 64, 3×3, stride=2) → ReLU
Decoder
  ConvTranspose2D(64 → 32, 3×3, stride=2) → ReLU
  ConvTranspose2D(32 → 1, 3×3, stride=2, output_padding=1) → Tanh
```

- **Latent space**: a 7×7×64 tensor encodes the compressed representation.
- **Activation**: we normalise images to [−1, 1] so a `tanh` decoder output matches the data distribution.
- **Loss**: mean squared error (MSE) between the original and reconstructed images.
- **Monitoring**: we log both reconstruction loss and PSNR (Peak Signal-to-Noise Ratio) to capture perceptual fidelity.

---

## 4. Training flow (mirrors other modules)

1. **Load Fashion-MNIST** (grayscale, 28×28).
2. **Normalise** to [−1, 1] for stable optimisation.
3. **Train** using Adam/AdamW, track validation loss + PSNR.
4. **Persist** best weights and training history to `artifacts/`.
5. **Reconstruct** samples via the inference helpers.

The notebooks present a guided walkthrough; the `src/` packages expose importable functions for automation, unit tests, or FastAPI integration.

---

## 5. Directory layout

```
pytorch/
  notebooks/deconv_pytorch.ipynb
  src/
    config.py      # device detection, hyperparameters, artifact dirs
    data.py        # dataset loading & dataloaders with paired targets
    model.py       # convolutional autoencoder (transposed conv decoder)
    engine.py      # train/eval loops computing loss + PSNR
    train.py       # CLI-style training entry point
    inference.py   # load checkpoint & reconstruct tensors
    utils.py       # PSNR, formatting, metric helpers

tensorflow/
  notebooks/deconv_tensorflow.ipynb
  src/
    config.py
    data.py        # tf.data pipeline emitting (image, image)
    model.py       # Keras autoencoder with Conv2DTranspose layers
    utils.py       # compile helper, metrics persistence
    train.py
    inference.py
```

Artefacts are written to `artifacts/pytorch_deconv/` and `artifacts/tensorflow_deconv/`.

---

## 6. Learning checklist

- [ ] Sketch the encoder/decoder shapes on paper to ensure spatial sizes match up.
- [ ] Print intermediate activations in the notebook to verify the upsampling behaviour.
- [ ] Experiment with kernel sizes, strides, or `output_padding` to see how the reconstruction quality changes.
- [ ] Compare PSNR between PyTorch and TensorFlow runs; ensure device choice (MPS/GPU/CPU) is reflected in performance.

---

## 7. Where to go next

Once you are comfortable with transposed convolutions, you are ready to tackle:

- **Segmentation networks** (U-Net, SegNet) that stack multiple upsampling stages.
- **GAN generators** (DCGAN, StyleGAN) which use progressive transposed convolutions to synthesise images.
- **Super-resolution models** (ESPCN, SRGAN) that upscale low-resolution inputs.
- **Decoder blocks in diffusion or transformer pipelines** where learned upsampling is combined with attention or skip connections.

This module sets the foundation—subsequent folders in the Deep Learning tree will extend these ideas with richer architectures.
