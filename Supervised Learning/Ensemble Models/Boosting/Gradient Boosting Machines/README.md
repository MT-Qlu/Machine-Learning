# Gradient Boosting Machines

The canonical gradient boosting models from scikit-learn live here. They provide a strong baseline for both regression and classification tasks before moving to more exotic boosters.

---

## Theory in brief

Gradient boosting fits each weak learner to the negative gradient of the loss:

$$
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)
$$

The learning rate $\nu$ controls the step size and regularises the ensemble.

## When to use

- Strong baselines on structured/tabular data.
- You need a balance between accuracy and interpretability.
- A reference point before using XGBoost or LightGBM.

## Beginner example

Imagine predicting taxi fares. The first small tree makes a rough guess, the next tree learns the errors of that guess, and the next corrects the remaining mistakes. After many small corrections, the final prediction becomes much more accurate.

Structure:

- `Classification/` — end-to-end pipeline for discrete targets with explainability hooks and probability calibration.
- `Regression/` — companion workflow for continuous targets featuring residual plots and interval estimates.

Each workflow keeps consistent folder semantics:

- `src/` stores configuration, feature engineering, training, and inference services.
- `data/` documents dataset ingestion and caching.
- `notebooks/` reproduces experiments with narrative context.
- `artifacts/` contains persisted models and evaluation outputs consumed by FastAPI.

Use this directory to compare learning-rate schedules, tree depth, and subsampling strategies across boosting flavours. Document noteworthy experiments in the notebooks so future iterations have a reliable baseline.

## Workflow tips

- Start with shallow trees and moderate learning rate (0.05–0.1).
- Use validation curves to find the sweet spot for `n_estimators`.
- Track feature importance to sanity-check model behaviour.
