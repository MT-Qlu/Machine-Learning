<!-- markdownlint-disable MD013 -->
# Support Vector Machine Modules

This directory hosts the support vector family of models, grouped into dedicated classification and regression submodules. Each submodule mirrors the repository-wide structure with reproducible training scripts, persisted artefacts, exploratory notebooks, and FastAPI-ready inference services.

---

## Theory in brief

SVMs find a maximum-margin separating hyperplane; kernels allow non-linear boundaries:

$$
\min_{w, b} \frac{1}{2} \lVert w \rVert^2 \quad \text{s.t.} \quad y_i (w^{\top} x_i + b) \geq 1
$$

The **RBF kernel** is common for non-linear data, while linear SVMs scale better to large datasets.

## When to use

- Medium-sized datasets with clear margins.
- High-dimensional feature spaces (text, TF-IDF).
- You need strong accuracy with careful tuning.

## Beginner example

Think of classifying emails as “spam” or “not spam.” An SVM tries to draw the widest possible gap between the two groups, so future emails that fall clearly inside one side are classified confidently. If the gap is not possible in the original space, a kernel lets the model bend the boundary to separate the groups.

## Submodules

- `Classification/` — Breast cancer diagnosis using an RBF-kernel SVC with probability calibration and feature scaling.
- `Regression/` — California housing price prediction via RBF-kernel SVR with quantile-aware evaluation and feature importances.

Refer to the README inside each submodule for datasets, CLI commands, notebooks, and extension ideas.

## Workflow tips

- Always scale features.
- Tune `C` and `gamma` with grid search.
- Use probability calibration when downstream decisions need calibrated scores.
