# AdaBoost Pipelines

Adaptive Boosting (AdaBoost) trains shallow learners sequentially, up-weighting the mistakes at each step. This directory holds both classification and regression flavours so you can compare loss surfaces and deployment behaviour side by side.

- `Classification/` targets categorical outcomes with stump-based learners and probability calibration.
- `Regression/` optimises absolute and squared losses for continuous targets, emphasising robustness to outliers.

Within each child directory you will find:

- `data/` — dataset downloaders or cached samples documented per project.
- `src/` — configuration, training loops, and FastAPI service hooks.
- `notebooks/` — exploratory analysis that mirrors the scripted run.
- `artifacts/` — persisted models, metrics, and schema files.

Use this folder as the launching point when extending AdaBoost with custom base estimators or alternative loss functions.
