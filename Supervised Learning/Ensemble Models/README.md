# Ensemble Learning Suite

**Location:** `Machine-Learning/Supervised Learning/Ensemble Models`

Ensemble methods combine multiple base learners to deliver stronger generalisation than any individual model. This suite mirrors the repository’s standard project structure (data, src, notebooks, artifacts, demo) so you can train, evaluate, and serve bagging and boosting pipelines alongside other supervised modules.

---

## Why ensembles work

- **Bagging** reduces variance by averaging many high-variance models (e.g., trees).
- **Boosting** reduces bias by sequentially correcting residual errors.
- **Diversity** across learners improves robustness and generalisation.

## When to use

- Tabular datasets with non-linear feature interactions.
- You need strong baselines with minimal feature engineering.
- Model stability and performance matter more than interpretability.

## Beginner example

Imagine asking 50 different interns to estimate a house price. Each intern makes mistakes, but the average of their guesses is usually closer to the truth than any single guess. That’s the intuition behind ensembles: many weak models combine into a stronger one.

## Structure

- `Bagging/` — Bootstrap aggregating and random forest implementations.
- `Boosting/` — Gradient boosting, stochastic GBM, AdaBoost, and XGBoost variants.

Each submodule includes its own README with dataset details, CLI commands, and FastAPI integration tips.

## Shared workflow

1. Train baseline models using the provided `train.py` scripts.
2. Compare cross-validated metrics and inspect feature importance.
3. Persist artefacts and expose FastAPI endpoints for reproducible inference.
4. Iterate on hyperparameters (n_estimators, depth, learning rate, subsampling).
