# Contractive Autoencoder

Promotes robustness to local perturbations by penalising the sensitivity of the encoder. The directory mirrors prior modules with PyTorch and TensorFlow implementations.

---

## Theory primer

Contractive autoencoders add a penalty on the **Jacobian of the encoder** to make the representation stable under small input changes:

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \|\nabla_{\mathbf{x}} f_\theta(\mathbf{x})\|_F^2
$$

This encourages **locally invariant features**, improving robustness to small perturbations and often producing smoother latent manifolds. The regulariser can be computed analytically for common activation functions or approximated via autodiff.

## Beginner example

If you slightly blur or shift a photo, you still want the representation to look similar. The contractive autoencoder penalises large changes in the encoding, so small input tweaks don’t cause big jumps in the latent space.

- `pytorch/` — Torch module that adds a Jacobian-based penalty inside the training loop.
- `tensorflow/` — Keras model with a custom training step implementing the same penalty.

Try comparing latent traversals between the vanilla and contractive variants to see how the penalty shapes the embedding geometry.

---

## Learning goals

- Investigate how contractive penalties smooth latent spaces and resist small adversarial perturbations.
- Compare reconstruction fidelity against the penalty magnitude across frameworks.
- Design experiments that vary penalty strength to diagnose under- or over-constraining the encoder.

---

## Implementation highlights

- Both frameworks share a config-driven layout so you can port experiments between them quickly.
- Penalty terms are exposed via helpers/metrics, making it easy to log in notebooks or dashboards.
- Training scripts export history JSONs you can analyse to study the trade-off between robustness and reconstruction.