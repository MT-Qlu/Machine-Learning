# Gradient Boosting Machines

The canonical gradient boosting models from scikit-learn live here. They provide a strong baseline for both regression and classification tasks before moving to more exotic boosters.

Structure:

- `Classification/` — end-to-end pipeline for discrete targets with explainability hooks and probability calibration.
- `Regression/` — companion workflow for continuous targets featuring residual plots and interval estimates.

Each workflow keeps consistent folder semantics:

- `src/` stores configuration, feature engineering, training, and inference services.
- `data/` documents dataset ingestion and caching.
- `notebooks/` reproduces experiments with narrative context.
- `artifacts/` contains persisted models and evaluation outputs consumed by FastAPI.

Use this directory to compare learning-rate schedules, tree depth, and subsampling strategies across boosting flavours. Document noteworthy experiments in the notebooks so future iterations have a reliable baseline.
