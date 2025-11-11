# XGBoost Workflows

These projects demonstrate Extreme Gradient Boosting (XGBoost) for structured data. The layout mirrors the rest of the supervised catalogue so training, evaluation, and serving stay uniform.

- `Classification/` delivers a gradient boosted classifier with feature importance inspection and FastAPI support.
- `Regression/` provides a regression counterpart tuned for tabular targets and interval forecasts.

Common expectations:

- Configuration and training logic reside in `src/`.
- Datasets or cached downloads live in `data/` alongside a README explaining provenance.
- Experiment notebooks reproduce the scripted runs and store visual diagnostics.
- Serialised models, metrics, and schema artefacts are written to `artifacts/` for downstream services.

Update the README in each child directory when you add new evaluation metrics, feature pipelines, or deployment options.
