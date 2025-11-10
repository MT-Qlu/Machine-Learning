# Unsupervised Learning Roadmap

Fast overview of the unsupervised modules currently scaffolded in this workspace. Each directory mirrors the production-ready structure used by supervised learning so future implementations can plug into notebooks, FastAPI services, and benchmarking pipelines without friction.

## K-Means Clustering
- **Core idea:** Partition observations into `k` clusters by minimising within-cluster variance.
- **Status:** Pipeline, training, inference, and CLI stubs created; notebooks and evaluation scripts forthcoming.
- **Next:** Add elbow/silhouette heuristics, scaling strategies, and FastAPI integration.

## DBSCAN
- **Core idea:** Discover arbitrarily shaped clusters based on density connectivity.
- **Status:** Scaffolding mirrors supervised layout with config-driven experiments and artefact persistence.
- **Next:** Provide parameter sweep notebooks and density visualisations.

## Gaussian Mixture Models
- **Core idea:** Model data as a weighted sum of Gaussian components using expectation-maximisation.
- **Status:** Pipeline and CLI placeholders ready; supports soft assignment exports via inference helpers.
- **Next:** Add BIC/AIC model selection notebook and probability-based evaluation utilities.

## Independent Component Analysis
- **Core idea:** Recover statistically independent sources from linear mixtures.
- **Status:** FastICA scaffolding in place with artefact persistence for decomposed signals.
- **Next:** Introduce source reconstruction notebooks and quality metrics.

## Principal Component Analysis
- **Core idea:** Project data onto orthogonal directions that maximise variance.
- **Status:** Transformer scaffolding and CLI stub available; ready for explained-variance analyses.
- **Next:** Add feature pre-processing notebooks and integration with supervised pipelines.

## Anomaly Detection
- **Core idea:** Score outliers via isolation-based models (with a path for One-Class SVM and LOF).
- **Status:** Isolation Forest scaffolding and CLI entry point ready; evaluation hooks forthcoming.
- **Next:** Add labelled benchmark datasets, ROC-style notebooks, and FastAPI inference.

## Time Series Analysis
- **Core idea:** Perform unsupervised diagnostics (seasonality decomposition, autocorrelation studies).
- **Status:** Seasonal decomposition pipeline scaffolded with artefact persistence.
- **Next:** Build notebooks for stationarity checks, feature engineering, and anomaly signalling.

---
**Tip:** Each module includes `data/`, `src/`, `artifacts/`, `notebooks/`, and `demo.py` just like the supervised side. Start in the directory README for setup notes, then dive into notebooks once they land.
