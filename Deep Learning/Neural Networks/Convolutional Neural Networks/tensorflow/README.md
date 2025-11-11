# TensorFlow CNN Module

TensorFlow/Keras counterpart to the PyTorch baseline. Use it to practice the same ideas with a different ecosystem.

---

## 1. Notebook walkthrough

- `notebooks/cnn_tensorflow.ipynb` mirrors the steps from the PyTorch notebook:
	- add `../src` to `sys.path`;
	- run `train.train()` to produce metrics and checkpoints;
	- sample predictions using the exported Keras model.
- Pay attention to how the `tf.data` pipeline replaces PyTorch dataloaders and how checkpoints are written in `.h5` format.

---

## 2. Source package tour

| File | Responsibility |
| ---- | -------------- |
| `config.py` | directories, batch size, epochs, and GPU/MPS visibility |
| `data.py` | loads Fashion-MNIST and constructs performant `tf.data.Dataset` objects |
| `model.py` | defines the Keras `Model` graph |
| `utils.py` | model compilation + history persistence |
| `train.py` | scripted training with built-in checkpoint callback |
| `inference.py` | loads the saved model and predicts class IDs |

Everything follows the same naming conventions as the rest of the Machine Learning repository, so switching contexts is painless.

---

## 3. Run the script

```bash
python -m pip install tensorflow-macos tensorflow-metal
python "Deep Learning/Neural Networks/Convolutional Neural Networks/tensorflow/src/train.py"
```

- Artefacts saved to `artifacts/tensorflow_cnn/`.
- The device helper prioritises Apple Silicon (MPS), then general GPU, finally CPU.

---

## 4. Practice prompts

1. Swap the optimiser (e.g., `tf.keras.optimizers.SGD`) and observe how convergence changes.
2. Add data augmentation inside `data.py` and compare validation accuracy.
3. Export the model to TensorFlow Lite and test inference on a mobile device or an embedded board.

These exercises keep the learning experience in sync with the PyTorch track while preparing you for more complex architectures.
