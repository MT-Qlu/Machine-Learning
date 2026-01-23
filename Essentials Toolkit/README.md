# Essentials Toolkit

**Location:** `Machine-Learning/Essentials`

This workspace collects reusable building blocks that support every pipeline in the repository. Each subdirectory includes its own README so you always know where you are and what utilities live there before dropping into code.

## Beginner example

If you’re training your first model, you can grab a loss metric from `Errors/`, scale features using `scaling/`, and compare experiments with the `Benchmark Tools/` configs—without re-writing common utilities.

## Directory Guide

- `Errors/` — legacy-compatible error metric definitions and helpers.
- `metrics/` — canonical loss and evaluation utilities for regression, classification, and forecasting.
- `optimizers/` — NumPy implementations of common optimisation algorithms for custom training loops.
- `scaling/` — feature scaling and normalisation routines with inverse-transform support.
- `Benchmark Tools/` — benchmarking runbooks, configuration stubs, and metric catalogues shared across model families.

Follow the breadcrumbs in each folder to continue deeper until you reach the implementation modules.
