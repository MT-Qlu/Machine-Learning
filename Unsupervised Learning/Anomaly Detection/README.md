# Anomaly Detection — Placeholder Module

This workspace will consolidate unsupervised anomaly detection techniques (Isolation Forest to start, with plans for One-Class SVM, Local Outlier Factor, and autoencoder variants). The directory mirrors the supervised-learning layout so future additions can plug directly into downstream APIs and experiment tracking.

---

## Learning Objectives

- Understand contamination-driven anomaly scoring and how isolation-based models rank points.
- Capture a consistent workflow for loading tabular datasets, training detectors, and persisting artefacts.
- Provide FastAPI-ready inference helpers and CLI demos akin to the supervised modules.

---

## Quickstart (Scaffolding)

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Drop raw data** into `data/` (CSV expected for now).
3. **Train the placeholder detector**

   ```bash
   python demo.py
   ```

4. **Inspect artefacts** in `artifacts/` (model + metrics notebooks to follow).

---

## Mathematical Foundations

Isolation Forest estimates anomaly scores by measuring the path length needed to isolate each observation in a random tree ensemble:

$$
\mathbb{E}[h(x)] = \frac{1}{t} \sum_{i=1}^{t} h_i(x)
$$

```
E[h(x)] = (1 / t) * sum_{i=1..t} h_i(x)
```

Shorter average path lengths signal anomalous samples because they are easier to isolate. Scores are normalised using the average path length of unsuccessful searches in binary trees, producing a decision function in `[−1, 1]` where negative values indicate potential anomalies.

---

## Roadmap

- [ ] Wire notebook walkthrough showcasing contamination tuning and ROC-style evaluation when labels exist.
- [ ] Integrate additional detectors (One-Class SVM, LOF) with a shared interface.
- [ ] Expose FastAPI endpoints mirroring the supervised registry pattern.

---

## Repository Layout

- `data/` — raw datasets (CSV for now).
- `src/` — modular pipeline, training, and inference utilities.
- `artifacts/` — persisted models, metrics, and plots.
- `notebooks/` — exploratory analysis and benchmarking (to be populated).
- `demo.py` — CLI entry point for quick experiments.
