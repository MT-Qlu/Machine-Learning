# Time Series Forecasting

This module collects classical forecasting strategies that power the time-series branch of the supervised learning catalogue. Each subfolder provides an end-to-end workflow with shared conventions—ingestion helpers in `data/`, training code in `src/`, exploratory notebooks in `notebooks/`, and lightweight artefacts in `artifacts/`.

---

## When to use classical forecasting

- You have **univariate** or lightly multivariate series with interpretable seasonality/trend.
- You need **fast baselines** before deploying heavier ML/DL models.
- Domain users want **transparent components** (trend, seasonality, residuals).

## Core concepts

- **Trend**: long-term direction (up/down growth).
- **Seasonality**: repeating cycles (monthly, weekly, yearly).
- **Residuals**: short-term noise or shocks.

The included models vary in how explicitly they model these components:

- ARIMA/SARIMA focus on autocorrelation and differencing.
- Prophet decomposes trend + seasonality + holidays.
- Exponential smoothing updates components with exponential weights.

## Common metrics

- **MAE**: robust to outliers.
- **RMSE**: penalises large errors.
- **MAPE**: easy to interpret (percentage), but unstable with zeros.

## Practical workflow

1. Plot the series and inspect seasonality/trend visually.
2. Start with a simple baseline (Exponential Smoothing or ARIMA).
3. Add seasonality (SARIMA/Prophet) if cycles are strong.
4. Validate with a chronological holdout or rolling-origin evaluation.
5. Persist artefacts and expose FastAPI endpoints for downstream use.

## Beginner example

Imagine a coffee shop tracking daily sales. You notice higher sales on weekends and a slow upward trend over months. A simple forecasting model learns the weekly pattern and the upward drift, then predicts next month’s sales so you can plan inventory without guessing.

## Included Workflows

- **ARIMA** — AutoRegressive Integrated Moving Average with seasonal extensions for tabular metrics and diagnostics.
- **SARIMA** — Seasonal ARIMA tuned for periodic demand signals and multi-step horizons.
- **Prophet** — Decomposable trend-plus-seasonality modelling using Meta's Prophet implementation.
- **Exponential Smoothing** — Holt-Winters style damped trend models for fast baselines.

## How to Use This Module

1. Enter the desired subfolder and review its local `README.md` for dataset, environment, and training details.
2. Run the `src/train.py` script to regenerate artefacts or adapt the pipeline to new data.
3. Surface the resulting model through the shared FastAPI service by registering the slug, mirroring the other supervised projects.

## Conventions

- Keep synthetic or public datasets small and documented inside the `data/` folder README.
- Record experiment context inside the notebooks and use markdown cells for design notes.
- Persist only reproducible artefacts (serialised models, metrics, configuration) to Git.
- Update this index when new forecasting techniques are added so the catalogue stays discoverable.
