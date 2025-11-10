# Optimisation Algorithms Reference

**Location:** `Machine-Learning/Essentials/optimizers`

This directory curates classic optimisation routines used across machine-learning pipelines. Each algorithm pairs mathematical intuition with a NumPy-backed implementation in `Essentials/optimizers/algorithms.py` so you can plug them into custom training loops or experiments.

## Included Optimisers

1. **Gradient Descent (Batch)** — deterministic updates using the full dataset.
2. **Stochastic Gradient Descent (SGD)** — noisy single-sample updates for online learning.
3. **Mini-Batch Gradient Descent** — hybrid approach balancing convergence stability and speed.
4. **Momentum & Nesterov Momentum** — accelerates gradients along low-curvature directions.
5. **AdaGrad** — adapts step sizes based on historical gradient magnitudes.
6. **RMSProp** — decays squared gradient history for non-stationary objectives.
7. **Adam** — combines momentum with RMSProp-style adaptive learning rates.

Each function exposes explicit parameters for learning rate, decay factors, and gradient callbacks, keeping the API consistent and easily testable.

## Quickstart

```python
import numpy as np
from Essentials.optimizers.algorithms import adam

params = np.zeros_like(gradients)
state = None
for step in range(1, 501):
    gradients = compute_gradients(params)
    params, state = adam(params, gradients, state=state)
```

- Pass gradients as NumPy arrays; utilities broadcast across shapes.
- Stateful optimisers (momentum-style) return an updated state dict you should feed back on the next iteration.
- Custom learning-rate schedules can be injected by modifying the `lr` argument per step.

## When to Reach for What

- **Convex, smooth objectives:** Start with vanilla gradient descent or momentum.
- **Sparse gradients / NLP:** AdaGrad and Adam handle varying magnitudes effectively.
- **Non-stationary objectives:** RMSProp and Adam adapt quickly as gradients shift.
- **Tiny learning rates / plateaus:** Nesterov momentum often escapes flat regions faster.

Future additions will cover second-order approximations (L-BFGS), adaptive clipping, and distributed optimisation templates.
