# Convolutional Neural Networks (CNN)

> **Goal**: learn the core convolutional network pattern that underpins most vision, audio, and spatial modelling workloads.

This module mirrors the style used in the **Supervised Learning** tree: a friendly theory primer, followed by runnable code in PyTorch and TensorFlow (notebook + modular package). Start here if you want to build intuition before exploring modern specialised models.

---

## 1. Why CNNs?

- **Local patterns matter**: neighbouring pixels carry correlated information (edges, textures). Convolutions exploit this by learning small filters that slide across the image.
- **Weight sharing**: the same filter is reused spatially, dramatically reducing parameters compared with fully connected layers.
- **Translation awareness**: because the filter is applied everywhere, the network recognises features regardless of where they appear.

Together, these properties make CNNs the default baseline for 2D images, spectrograms, and other grid-like signals.

---

## 2. Architecture at a glance

```
input (28×28×1)
 → Conv2D (3×3 kernel) + ReLU + BatchNorm
 → MaxPool (2×2) ↓ spatial size
 → Conv2D (3×3) + ReLU + BatchNorm
 → MaxPool (2×2)
 → Flatten → Dense (ReLU) → Dropout → Dense (logits)
```

- **Convolution**: a kernel \( W \) slides over the input \( X \). For one location `(i, j)`,
  \[
  (X * W)_{i,j} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} X_{i+u, j+v} · W_{u,v}
  \]
- **Pooling**: summarises small windows (e.g., max) to introduce robustness to small shifts.
- **Batch Normalisation**: stabilises training by normalising activations inside the network.
- **Dropout**: randomly drops activations (here inside the dense layers) to reduce overfitting.

---

## 3. Training pipeline (mirrors other modules)

1. **Dataset prep**
   - Fashion-MNIST (10 clothing classes, 28×28 grayscale).
   - Normalisation: scale to [−1, 1] so gradients behave nicely.
2. **Model definition**
   - PyTorch: `nn.Module` in `model.py`.
   - TensorFlow: `tf.keras.Model` in `model.py`.
3. **Optimisation**
   - Loss: cross entropy.
   - Optimiser: AdamW (PyTorch) / Adam (TensorFlow).
   - Device selection: MPS → CUDA → CPU (automatic).
4. **Evaluation**
   - Accuracy monitored per epoch; best weights are persisted to `artifacts/...`.
5. **Inference**
   - Dedicated helper loads checkpoints and returns predicted class indices.

The notebook in each framework demonstrates the complete flow (training log + inference sample) while the `src/` package is importable for automation or serving.

---

## 4. Directory layout

```
pytorch/
  notebooks/cnn_pytorch.ipynb      ← guided run in notebook form
  src/
    config.py                      ← hyperparameters + device detection
    data.py                        ← dataset download & dataloaders
    model.py                       ← CNN architecture
    engine.py                      ← train/eval loops
    train.py                       ← scriptable entry point
    inference.py                   ← checkpoint loading & predictions
    utils.py                       ← accuracy/time helpers

tensorflow/
  notebooks/cnn_tensorflow.ipynb
  src/
    config.py                      ← directory layout + TF device config
    data.py                        ← tf.data pipeline
    model.py                       ← Keras model definition
    train.py                       ← scripted training with callbacks
    inference.py                   ← `.h5` loading + predictions
    utils.py                       ← compile + metrics persistence
```

All artefacts live under `artifacts/pytorch_cnn/` and `artifacts/tensorflow_cnn/` for easy comparison.

---

## 5. Learning checklist

- [ ] Read through the notebook to see the training curves and sample predictions.
- [ ] Trace the forward pass in `model.py` to solidify tensor shapes after each layer.
- [ ] Experiment: adjust kernel sizes, add/drop layers, or change pooling and observe metrics.
- [ ] Swap Fashion-MNIST with another dataset (e.g., CIFAR-10) by editing `data.py` and updating the input shape.

When you feel comfortable, this same structure can be ported to specialised architectures or integrated into the `fastapi_app` project.

---

## 6. Beyond the vanilla CNN

The code here stops at the foundational stack. Modern research and production systems frequently rely on:

- **Residual / Skip connections** (ResNet, ResNeXt) for very deep models.
- **Dense connectivity** (DenseNet) to reuse features across layers.
- **Mobile-friendly networks** (MobileNetV3, EfficientNet, ConvNeXt) for edge devices.
- **Attention-based models** (Vision Transformers, Swin Transformer) that replace convolutions entirely.
- **Hybrid schemes** (CNN + Transformer, SE blocks, depthwise separable convolutions) for specialised use cases.

These variants will appear under their respective folders in the Deep Learning tree. Use the preparation here as the conceptual baseline before diving into those upgrades.
