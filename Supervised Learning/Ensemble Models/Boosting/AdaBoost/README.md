# AdaBoost Pipelines

Adaptive Boosting (AdaBoost) trains shallow learners sequentially, up-weighting the mistakes at each step. This directory holds both classification and regression flavours so you can compare loss surfaces and deployment behaviour side by side.

---

## Theory in brief

AdaBoost builds an additive model of weak learners:

$$
F_m(x) = F_{m-1}(x) + \alpha_m h_m(x)
$$

Samples misclassified at step $m-1$ receive higher weights so the next learner focuses on hard cases.

## When to use

- You want a strong baseline with minimal feature engineering.
- Small to medium tabular datasets.
- Interpretable weak learners (stumps) are acceptable.

- `Classification/` targets categorical outcomes with stump-based learners and probability calibration.
- `Regression/` optimises absolute and squared losses for continuous targets, emphasising robustness to outliers.

Within each child directory you will find:

- `data/` — dataset downloaders or cached samples documented per project.
- `src/` — configuration, training loops, and FastAPI service hooks.
- `notebooks/` — exploratory analysis that mirrors the scripted run.
- `artifacts/` — persisted models, metrics, and schema files.

Use this folder as the launching point when extending AdaBoost with custom base estimators or alternative loss functions.

## Workflow tips

- Start with decision stumps to avoid overfitting.
- Tune `n_estimators` and `learning_rate` jointly.
- Monitor training/validation curves for instability or noise amplification.
