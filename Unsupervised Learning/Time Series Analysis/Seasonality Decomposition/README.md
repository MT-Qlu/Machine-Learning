# Seasonality Decomposition

This directory houses tooling for separating seasonal patterns from time-series signals. Use it to prototype additive or multiplicative decompositions prior to forecasting or anomaly detection.

## Contents

- `src/` — implementations of classical decomposition, STL, and helper utilities for seasonal strength diagnostics.
- `notebooks/` — hands-on walkthroughs illustrating how seasonal components evolve across datasets.
- `data/` — curated sample series that exhibit seasonal behaviour; document their origins for reproducibility.
- `artifacts/` — cached decomposed components, plots, or configuration snapshots consumed by downstream modules.

Keep this README current when new decomposition techniques or evaluation recipes are added.
