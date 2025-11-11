# Stochastic Gradient Boosting

This implementation augments gradient boosting with stochastic subsampling to improve generalisation on structured datasets. Two sibling pipelines live under this directory:

- `Classification/` tackles tabular classification problems.
- `Regression/` targets continuous targets with pinball metrics and residual diagnostics.

Both variants share:

- Data preparation helpers inside `data/` and `src/data.py`.
- Training entry points (`src/train.py`) that manage hyperparameters, evaluation, and artefact persistence.
- Notebooks that replicate the scripted workflow while exposing experiment notes.
- Lightweight artefacts used by the FastAPI services for reproducible inference.

Tune the learning rate, subsampling ratios, and tree depth in the configuration module before re-running training. Update the FastAPI registry when a new model version is ready to ship.
