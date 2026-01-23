# Random Forest Workflows

Random Forests form the baseline bagging ensemble for both classification and regression tasks. Two sibling projects live here:

- `Classification/` — trains a forest on categorical targets with feature importance reporting and inference helpers.
- `Regression/` — adapts the same tooling for continuous targets with interval estimation and error analysis.

Common layout expectations:

- `src/` contains configuration, data loading, training loops, and FastAPI integration modules.
- `data/` documents dataset origins and caching so the pipelines remain reproducible.
- `notebooks/` recreates the training run with richer exploratory analysis and visual diagnostics.
- `artifacts/` stores persisted models, metrics, and schemas consumed by downstream services.

Use this README as the entry point when refreshing hyperparameters, experimenting with feature subsets, or benchmarking against other bagging variants.

---

## Theory in brief

Random Forests combine bagging with **feature subsampling** at each split, yielding de-correlated trees:

$$
\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

This lowers variance without significantly increasing bias.

## When to use

- Strong baseline on tabular data with mixed feature types.
- You want feature importance estimates and stable predictions.
- You need robust performance without extensive feature engineering.

## Beginner example

Imagine asking 100 people to guess the price of a house, but each person only sees a random subset of features. Some guesses are off, but the average is stable. That’s how a random forest produces reliable predictions without heavy tuning.

## Workflow tips

- Tune `n_estimators`, `max_depth`, and `max_features` first.
- Use OOB score to approximate validation performance quickly.
- Inspect feature importances and permutation importance for interpretability.
