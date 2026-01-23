# Regularised Linear Models Suite

**Location:** `Machine-Learning/Supervised Learning/Regularized Linear Models`

This collection extends the linear regression baseline with L2, L1, and elastic-net penalties. Use these modules when multicollinearity, high-dimensional feature spaces, or sparse solutions matter. Each subdirectory mirrors the standard repository structure so you can train, evaluate, and serve models consistently.

---

## Theory in brief

Regularisation adds a penalty to the OLS objective:

$$
\mathcal{L} = \|y - X\beta\|_2^2 + \lambda \mathcal{P}(\beta)
$$

- **Ridge (L2)**: $\mathcal{P}(\beta) = \|\beta\|_2^2$ (shrinks coefficients).
- **Lasso (L1)**: $\mathcal{P}(\beta) = \|\beta\|_1$ (sparsifies coefficients).
- **Elastic Net**: combines L1 and L2 for stability + sparsity.

## When to use

- Many correlated features (multicollinearity).
- Need feature selection or sparse coefficients (Lasso/Elastic Net).
- Overfitting risk in high-dimensional settings.

## Beginner example

Suppose you’re predicting house prices from 200 features, many of which overlap. A regularised model shrinks noisy coefficients so the model doesn’t overreact to random correlations. Lasso might even drop unhelpful features entirely, leaving a simpler, more stable model.

## Modules

- `Ridge Regression/` — L2-penalised regression with coefficient shrinkage (fully scaffolded with code and dataset).
- `Lasso Regression/` — L1-penalised regression driving coefficients to zero for feature selection.
- `Elastic Net/` — Hybrid L1/L2 penalty balancing shrinkage and sparsity.

Follow the README inside each module for dataset notes, training commands, and integration guidance.

## Workflow tips

- Standardise features before fitting.
- Tune regularisation strength using cross-validation.
- Compare coefficient paths to understand feature sensitivity.
