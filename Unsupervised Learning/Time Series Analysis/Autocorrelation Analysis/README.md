# Autocorrelation Analysis

This area focuses on autocorrelation and partial autocorrelation diagnostics for time series. Use the tooling here to understand temporal dependence before fitting forecasting or anomaly models.

## Workflow Outline

- `src/` hosts helpers for computing ACF and PACF statistics, along with visualisation utilities.
- `notebooks/` walk through typical explorations, highlighting how to interpret lag structures and confidence intervals.
- `data/` records sample series used in the notebooks or automated tests; keep provenance documented.
- `artifacts/` captures cached figures or serialised statistics that downstream modules may reuse.

Extend this README when you introduce new diagnostics (Ljung-Box tests, spectral density plots, etc.) so future contributors know where to start.
