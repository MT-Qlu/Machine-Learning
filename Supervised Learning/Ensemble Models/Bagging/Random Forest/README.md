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
