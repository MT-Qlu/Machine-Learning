# Time Series Forecasting

This module collects classical forecasting strategies that power the time-series branch of the supervised learning catalogue. Each subfolder provides an end-to-end workflow with shared conventions—ingestion helpers in `data/`, training code in `src/`, exploratory notebooks in `notebooks/`, and lightweight artefacts in `artifacts/`.

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
