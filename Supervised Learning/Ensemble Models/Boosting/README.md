# Boosting Ensembles

Boosting stacks sequential weak learners so that each iteration focuses on the residual mistakes from the previous one. This hub documents the boosting families supported in the supervised catalogue. Every child directory follows the shared layout (`data/`, `src/`, `notebooks/`, `artifacts/`) to keep training and serving workflows consistent.

## Available Pipelines

- **AdaBoost** — Adaptive re-weighting of samples with decision stumps or shallow trees.
- **Gradient Boosting Machines** — scikit-learn gradient boosting for structured regression and classification.
- **Stochastic Gradient Boosting** — Learning rate tuned gradient boosting with row and column subsampling for regularisation.
- **XGBoost** — Extreme Gradient Boosting with histogram optimisations and fast inference.

Each pipeline includes both classification and regression variants where applicable. Check the nested `README.md` files for dataset choices, feature schemas, and FastAPI integration notes.

## Extending

When adding a new booster:

1. Create a sibling directory and mirror the folder layout used here.
2. Document training steps, hyperparameters, and evaluation protocol inside the new README.
3. Update the FastAPI registry once artefacts are persisted so the service can load the new model.
