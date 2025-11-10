<!-- markdownlint-disable MD013 -->
# Decision Tree Modules

This directory collects decision tree baselines for both classification and regression tasks. Each submodule follows the repository-standard structure with scripted training, persisted artefacts, exploratory notebooks, and FastAPI-ready inference services.

## Submodules

- `Classification/` — Iris species prediction using a depth-controlled `DecisionTreeClassifier` with probability outputs and feature importances.
- `Regression/` — California housing value estimation using a `DecisionTreeRegressor`, including feature importance reporting for interpretability.

Refer to the README inside each submodule for datasets, command examples, and extension ideas.
