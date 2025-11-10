# DBSCAN Clustering — Placeholder Module

This directory houses the scaffolding for a density-based spatial clustering (DBSCAN) workflow. The structure mirrors the supervised-learning layout so documentation, notebooks, and FastAPI hooks can drop in as the implementation matures.

---

## Learning Objectives

- Refresh the intuition behind density reachability and core/border/noise points.
- Provide a reproducible pipeline for fitting DBSCAN, capturing artefacts, and benchmarking hyperparameters (`eps`, `min_samples`).
- Prepare the codebase for future integrations such as HDBSCAN, OPTICS, and density visualisation notebooks.

---

## Quickstart (Scaffolding)

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place data** in `data/dataset.csv` (feature-only tabular format).
3. **Run the demo**

   ```bash
   python demo.py
   ```

   The script fits DBSCAN with placeholder defaults and persists the model under `artifacts/dbscan.joblib`.

---

## Mathematical Foundations

DBSCAN groups points that satisfy density connectivity. A point `p` is a core point if the `eps`-neighbourhood contains at least `min_samples` observations:

$$
\lvert N_{\varepsilon}(p) \rvert \geq \text{min\_samples}
$$

```
|N_eps(p)| >= min_samples
```

All points density-reachable from a core point belong to the same cluster. Points not reachable from any core point are labelled as noise (`-1`). Selecting `eps` controls the neighbourhood radius, while `min_samples` sets the minimum density threshold.

---

## Roadmap

- [ ] Notebook walkthrough illustrating parameter sweeps and silhouette/DBCV scores.
- [ ] Feature scaling and dimensionality reduction helpers to improve clustering quality.
- [ ] Integration with upcoming benchmarking/evaluation playbooks for batch experimentation.

---

## Repository Layout

- `data/` — raw feature tables for clustering.
- `src/` — pipeline, training, and inference helpers.
- `artifacts/` — persisted clustering models and evaluation exports.
- `notebooks/` — exploratory visualisations and diagnostics (to be populated).
- `demo.py` — CLI entry point for quick experimentation.
