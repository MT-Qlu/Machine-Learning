<!-- markdownlint-disable MD013 -->
# Decision Tree Modules

This directory collects decision tree baselines for both classification and regression tasks. Each submodule follows the repository-standard structure with scripted training, persisted artefacts, exploratory notebooks, and FastAPI-ready inference services.

---

## Theory in brief

Decision trees recursively split the feature space to minimise impurity:

- **Classification**: minimise Gini or entropy.
- **Regression**: minimise variance or mean squared error.

Trees are interpretable and naturally capture non-linear interactions, but they can overfit without constraints.

## When to use

- You need an interpretable baseline with feature importance.
- Relationships are non-linear and interaction-heavy.
- You want a fast model without heavy preprocessing.

## Submodules

- `Classification/` — Iris species prediction using a depth-controlled `DecisionTreeClassifier` with probability outputs and feature importances.
- `Regression/` — California housing value estimation using a `DecisionTreeRegressor`, including feature importance reporting for interpretability.

Refer to the README inside each submodule for datasets, command examples, and extension ideas.

## Workflow tips

- Start with shallow depths; increase only if validation metrics improve.
- Monitor overfitting by comparing train vs validation curves.
- Consider pruning or `min_samples_leaf` to stabilise predictions.
