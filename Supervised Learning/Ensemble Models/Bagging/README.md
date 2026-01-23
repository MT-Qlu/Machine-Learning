# Bagging Ensembles

Bagging (bootstrap aggregation) trains multiple estimators on resampled data and averages their predictions to stabilise variance. This directory groups the bagging-focused workflows in the supervised catalogue.

---

## Theory in brief

Bagging fits $B$ models on bootstrap samples and aggregates predictions:

$$
\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)
$$

This reduces variance and improves stability, especially for high-variance learners like decision trees.

## When to use

- High-variance models that overfit single training sets.
- You want strong performance with minimal tuning.
- Interpretability is less critical than accuracy.

## Current Contents

- `Random Forest/` — canonical bagging of decision trees for both classification and regression. The subdirectories follow the shared `data/`, `src/`, `notebooks/`, and `artifacts/` structure to keep orchestration straightforward.

When you introduce additional bagging strategies (extra trees, bagged linear models, etc.), mirror this layout and extend the FastAPI registry accordingly.

## Recommended workflow

1. Train a baseline single tree to set a variance benchmark.
2. Train bagged models with increasing `n_estimators`.
3. Compare out-of-bag (OOB) estimates with validation scores.
4. Persist artefacts for FastAPI inference and monitoring.
