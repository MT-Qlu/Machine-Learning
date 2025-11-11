# Bagging Ensembles

Bagging (bootstrap aggregation) trains multiple estimators on resampled data and averages their predictions to stabilise variance. This directory groups the bagging-focused workflows in the supervised catalogue.

## Current Contents

- `Random Forest/` — canonical bagging of decision trees for both classification and regression. The subdirectories follow the shared `data/`, `src/`, `notebooks/`, and `artifacts/` structure to keep orchestration straightforward.

When you introduce additional bagging strategies (extra trees, bagged linear models, etc.), mirror this layout and extend the FastAPI registry accordingly.
