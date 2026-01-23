# Stochastic Gradient Boosting

This implementation augments gradient boosting with stochastic subsampling to improve generalisation on structured datasets. Two sibling pipelines live under this directory:

---

## Theory in brief

Stochastic gradient boosting introduces row/column subsampling, which reduces variance and improves robustness:

$$
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \quad h_m \text{ fit on a subsample}
$$

Subsampling acts as regularisation and can reduce overfitting in noisy datasets.

## When to use

- Datasets with noise or risk of overfitting.
- You want a stronger generalising booster than vanilla GBM.
- You need a balance of speed and performance.

- `Classification/` tackles tabular classification problems.
- `Regression/` targets continuous targets with pinball metrics and residual diagnostics.

Both variants share:

- Data preparation helpers inside `data/` and `src/data.py`.
- Training entry points (`src/train.py`) that manage hyperparameters, evaluation, and artefact persistence.
- Notebooks that replicate the scripted workflow while exposing experiment notes.
- Lightweight artefacts used by the FastAPI services for reproducible inference.

Tune the learning rate, subsampling ratios, and tree depth in the configuration module before re-running training. Update the FastAPI registry when a new model version is ready to ship.

## Workflow tips

- Start with `subsample=0.8` and `max_features=0.8`.
- Use early stopping if available in the pipeline.
- Track residual plots to diagnose systematic errors.
