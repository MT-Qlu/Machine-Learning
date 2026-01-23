<!-- markdownlint-disable MD013 -->
# K-Nearest Neighbours Modules

This directory groups the K-Nearest Neighbours (KNN) implementations for both classification and regression tasks. Each subfolder mirrors the repository-wide supervised-learning structure: reproducible configuration, scripted training, persisted artefacts, FastAPI integration, exploratory notebooks, and demo scripts.

---

## Theory in brief

KNN predicts using the $k$ closest training samples under a distance metric:

- **Classification**: majority vote (optionally distance-weighted).
- **Regression**: average of neighbour targets.

Distance metrics and feature scaling strongly influence performance.

## When to use

- Small to medium datasets with meaningful local structure.
- Non-linear boundaries without model assumptions.
- Quick, interpretable baselines (neighbour inspection).

## Submodules

- `Classification/` — multi-class wine classification using distance-weighted KNN.
- `Regression/` — diabetes progression regression using distance-weighted KNN.

Refer to the README inside each submodule for detailed instructions, datasets, and extension ideas.

## Workflow tips

- Always scale features before fitting.
- Tune `k` and distance metrics with cross-validation.
- Use distance-weighted voting for smoother decision boundaries.
