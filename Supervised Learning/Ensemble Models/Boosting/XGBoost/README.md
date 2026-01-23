# XGBoost Workflows

These projects demonstrate Extreme Gradient Boosting (XGBoost) for structured data. The layout mirrors the rest of the supervised catalogue so training, evaluation, and serving stay uniform.

---

## Theory in brief

XGBoost optimises a regularised objective:

$$
\mathrm{Obj} = \sum_i l\big(y_i, \hat{y}_i^{(t)}\big) + \sum_k \Omega(f_k)
$$

It adds second-order Taylor approximations, shrinkage, and column subsampling to improve convergence and generalisation.

## When to use

- Strong tabular baselines with mixed feature types.
- You need high accuracy and robust feature importance.
- You can afford tuning for depth, learning rate, and subsampling.

## Beginner example

Think of a quiz where each new study session focuses only on the questions you missed last time. XGBoost does the same: each new tree fixes the mistakes of the previous ones, so accuracy keeps improving in small, controlled steps.

- `Classification/` delivers a gradient boosted classifier with feature importance inspection and FastAPI support.
- `Regression/` provides a regression counterpart tuned for tabular targets and interval forecasts.

Common expectations:

- Configuration and training logic reside in `src/`.
- Datasets or cached downloads live in `data/` alongside a README explaining provenance.
- Experiment notebooks reproduce the scripted runs and store visual diagnostics.
- Serialised models, metrics, and schema artefacts are written to `artifacts/` for downstream services.

Update the README in each child directory when you add new evaluation metrics, feature pipelines, or deployment options.

## Workflow tips

- Start with moderate depth (4–6) and learning rate (0.05–0.1).
- Use early stopping on a validation set to prevent overfitting.
- Track feature importance and SHAP for interpretability.
