<!-- markdownlint-disable MD013 -->
# Decision Tree Regression — California Housing

This module wraps a decision tree regressor for predicting median house values across California census block groups. It mirrors the repository-wide supervised-learning pattern: reproducible configuration, scripted training, persisted artefacts, FastAPI integration, notebook exploration, and a lightweight demo. Use it to benchmark non-linear regressors, showcase feature importances, or seed downstream ensemble experiments.

---

## Learning Objectives

- Understand how decision trees fit continuous targets using variance reduction.
- Load, cache, and explore the California housing dataset with consistent feature naming.
- Train, evaluate, and persist a `DecisionTreeRegressor` pipeline with depth and leaf controls.
- Serve predictions (with feature importance context) through the shared FastAPI registry.
- Extend the baseline with pruning, quantile evaluation, and ensemble enhancements.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Decision Tree/Regression/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "r2": 0.66,
     "rmse": 0.56,
     "mae": 0.41,
     "tree_depth": 12.0,
     "num_leaves": 1042.0
   }
   ```

   Artefacts are saved to `artifacts/decision_tree_regressor.joblib` and `artifacts/metrics.json`. The dataset is cached at `data/california_housing.csv` on the first run.

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a prediction** (after registering the `decision_tree_regression` slug)

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/decision_tree_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "median_income": 6.5,
              "house_age": 28.0,
              "average_rooms": 5.4,
              "average_bedrooms": 1.1,
              "population": 830.0,
              "average_occupancy": 3.2,
              "latitude": 34.1,
              "longitude": -118.2
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_value": 3.91,
     "feature_importances": {
       "median_income": 0.54,
       "house_age": 0.04,
       "average_rooms": 0.19,
       "average_bedrooms": 0.01,
       "population": 0.04,
       "average_occupancy": 0.14,
       "latitude": 0.03,
       "longitude": 0.01
     },
     "model_version": "1731264000",
     "metrics": {
       "r2": 0.66,
       "rmse": 0.56,
       "mae": 0.41,
       "tree_depth": 12.0,
       "num_leaves": 1042.0
     }
   }
   ```

5. **Open the companion notebook**

   `notebooks/decision_tree_regression.ipynb` reproduces the pipeline, evaluates metrics, and visualises feature importances.

6. **Optional Docker workflow**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles all supervised modules. On the first regression request the service trains automatically if artefacts are missing.

---

## Mathematical Foundations

Decision tree regression chooses splits that minimise the weighted mean squared error of child nodes. For a node $t$ with response variance $Var(t)$, a candidate split $s$ yields impurity reduction

$$
\Delta Var(s, t) = Var(t) - \frac{N_{left}}{N_t} Var(t_{left}) - \frac{N_{right}}{N_t} Var(t_{right}).
$$

The algorithm greedily selects the split with the largest variance reduction until stopping criteria are met. Predictions in each leaf are simply the mean target value of the training samples that fall into that leaf. Cost-complexity pruning (controlled via `ccp_alpha`) can reduce overfitting by penalising depth.

### Plain-Language Intuition

The tree repeatedly asks questions such as “Is median income > 6.4?” or “Is latitude ≤ 35?” to carve California into regions with similar house prices. Each split narrows the price range. By the time a block reaches a leaf, the remaining homes share similar characteristics, so their average price is a good prediction for new homes that land in the same leaf. Feature importances quantify which questions most reduced price variance across the dataset.

---

## Dataset

- **Source**: California housing dataset via `sklearn.datasets.fetch_california_housing` (20,640 observations, 8 numeric predictors).
- **Cache**: `data/california_housing.csv` (auto-generated on first run).
- **Target**: `median_house_value` (in $100k units).

Feature overview:

| Column              | Description                                   |
|---------------------|-----------------------------------------------|
| `median_income`     | Median household income (scaled by $10k)      |
| `house_age`         | Median house age in years                     |
| `average_rooms`     | Average rooms per household                   |
| `average_bedrooms`  | Average bedrooms per household                |
| `population`        | Block group population                        |
| `average_occupancy` | Average household size                        |
| `latitude`          | Geographic latitude                           |
| `longitude`         | Geographic longitude                          |

An 80/20 random split keeps evaluation consistent across notebook and CLI runs.

---

## Repository Layout

```
Regression/
├── README.md
├── artifacts/
│   └── .gitkeep
├── data/
│   └── california_housing.csv
├── demo.py
├── notebooks/
│   └── decision_tree_regression.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── inference.py
    ├── pipeline.py
    └── train.py
```

---

## Implementation Walkthrough

- `config.py` — captures file paths, hyperparameters, and directory bootstrapping.
- `data.py` — downloads the California housing dataset, renames columns, and produces train/validation splits.
- `pipeline.py` — trains the `DecisionTreeRegressor`, logs R²/RMSE/MAE plus structural diagnostics, and persists artefacts via Joblib/JSON.
- `inference.py` — defines Pydantic request/response schemas, surfaces feature importances, and plugs into the FastAPI registry.
- `train.py` — CLI entry point invoked by docs, CI, or manual retraining.
- `demo.py` — quick smoke test to verify the model and service contract.

---

## Extensions

1. **Pruning grid** — sweep `max_depth`, `min_samples_leaf`, and `ccp_alpha` to balance bias/variance.
2. **Quantile analysis** — evaluate absolute percentage error across price buckets for better monitoring signals.
3. **Hybrid features** — engineer ratios such as rooms-per-occupant or interact with latitude/longitude bins.
4. **Ensemble upgrade** — drop the same preprocessing into `RandomForestRegressor` or `GradientBoostingRegressor` under the Ensemble roadmap.
5. **Monitoring** — log residual distributions and feature importances in production to detect drift.

---

## References

- L. Breiman, J. Friedman, R. Olshen, C. Stone. *Classification and Regression Trees.*
- Trevor Hastie, Robert Tibshirani, Jerome Friedman. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.tree.DecisionTreeRegressor` and cost-complexity pruning.
