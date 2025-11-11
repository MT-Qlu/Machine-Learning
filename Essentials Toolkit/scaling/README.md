# Feature Scaling & Normalisation Reference

**Location:** `Machine-Learning/Essentials/scaling`

Consistent scaling prevents optimisation pitfalls and ensures distance-based models behave sensibly. This directory catalogues the most common data-transformation strategies alongside NumPy implementations in `Essentials/scaling/preprocessing.py`.

## Techniques Covered

1. **Standardisation (Z-Score)** — centre to zero mean and unit variance.
2. **Min-Max Scaling** — map features into a fixed `[0, 1]` or custom range.
3. **Robust Scaling** — use median and interquartile range to resist outliers.
4. **Max-Abs Scaling** — scale by the maximum absolute value, preserving sparsity.
5. **Unit Vector Normalisation** — project samples to the unit hypersphere for cosine/distance-based models.
6. **Log Scaling** — stabilise variance for positive-valued features.

Each transformer returns both the transformed data and the fitted statistics to support inverse transforms and deployment.

## Quickstart

```python
import numpy as np
from Essentials.scaling.preprocessing import standardise

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 40.0]])
X_scaled, params = standardise(X)
```

- Pass 2D arrays (`n_samples`, `n_features`). Functions broadcast across features.
- Persist the parameters dictionary to reproduce the same transformation at inference time.
- Use `inverse_transform_*` helpers to map predictions back into the original feature space.

Future updates will add pipeline utilities for chaining scalers and compatibility adapters for scikit-learn transformers.
