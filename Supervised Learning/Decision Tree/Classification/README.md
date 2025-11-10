<!-- markdownlint-disable MD013 -->
# Decision Tree Classification — Iris Species

This module provides a production-ready decision tree classifier for the classic Iris dataset. It follows the shared repository template: reproducible configuration, scripted training, persisted artefacts, FastAPI integration, an exploratory notebook, and a lightweight demo script. Use it to revisit tree-based classification, illustrate interpretability, or benchmark against distance and margin-based baselines.

---

## Learning Objectives

- Refresh the intuition behind axis-aligned tree splits, impurity metrics, and pruning knobs.
- Load, cache, and explore the Iris dataset with consistent feature naming.
- Train, evaluate, and persist a `DecisionTreeClassifier` pipeline with minimal boilerplate.
- Serve probability predictions (with feature importance context) through the shared FastAPI registry.
- Extend the baseline with depth limits, cost-complexity pruning, and ensemble experiments.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Decision Tree/Classification/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "accuracy": 0.967,
     "macro_precision": 0.969,
     "macro_recall": 0.967,
     "macro_f1": 0.967,
     "tree_depth": 5.0,
     "num_leaves": 9.0
   }
   ```

   Artefacts are saved to `artifacts/decision_tree_classifier.joblib` and `artifacts/metrics.json`. The dataset is cached at `data/iris.csv` on the first run.

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Submit a prediction** (after registering the `decision_tree_classification` slug)

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/decision_tree_classification" \
        -H "Content-Type: application/json" \
        -d '{
              "sepal_length_cm": 6.0,
              "sepal_width_cm": 3.1,
              "petal_length_cm": 4.8,
              "petal_width_cm": 1.6
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_label": "versicolor",
     "class_probabilities": {
       "setosa": 0.0,
       "versicolor": 1.0,
       "virginica": 0.0
     },
     "feature_importances": {
       "sepal_length_cm": 0.083,
       "sepal_width_cm": 0.0,
       "petal_length_cm": 0.416,
       "petal_width_cm": 0.501
     },
     "model_version": "1731264000",
     "metrics": {
       "accuracy": 0.967,
       "macro_precision": 0.969,
       "macro_recall": 0.967,
       "macro_f1": 0.967,
       "tree_depth": 5.0,
       "num_leaves": 9.0
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/decision_tree_classification.ipynb` to inspect the dataset, re-run training, and rank feature importances.

6. **Optional Docker workflow**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles every supervised module. On the first classification request the service trains automatically if artefacts are missing.

---

## Mathematical Foundations

Decision trees recursively partition the feature space. At each node, the algorithm selects the feature and threshold that yields the largest impurity reduction (Gini index by default):

$$
G(t) = 1 - \sum_{k=1}^{K} p_{k,t}^2,
$$

where $p_{k,t}$ is the proportion of class $k$ in node $t$. A split $s$ partitions the node into left/right children and maximises the impurity decrease

$$
\Delta G(s, t) = G(t) - \frac{N_{left}}{N_t} G(t_{left}) - \frac{N_{right}}{N_t} G(t_{right}).
$$

The process continues until stopping criteria are met (maximum depth, minimum samples per leaf, or pure nodes). Cost-complexity pruning can later shrink the tree by trading depth for generalisation using the $\alpha$ penalty configured via `ccp_alpha`.

### Plain-Language Intuition

A decision tree asks a series of yes/no questions that progressively isolate a class. For the Iris dataset, it might begin with “Is petal length ≤ 2.45 cm?” (separating setosa) before drilling into petal width and sepal length to distinguish versicolor from virginica. Each question slices the dataset into more homogeneous groups, and the final leaf votes for the most common species in that slice. Feature importances report how much each question helped reduce uncertainty across the tree.

---

## Dataset

- **Source**: Iris dataset via `sklearn.datasets.load_iris` (150 samples, 3 classes, 4 numeric features).
- **Cache**: `data/iris.csv` (auto-generated on first run).
- **Target**: `species` with labels `{setosa, versicolor, virginica}`.

Feature overview:

| Column             | Description                            |
|--------------------|----------------------------------------|
| `sepal_length_cm`  | Sepal length in centimetres            |
| `sepal_width_cm`   | Sepal width in centimetres             |
| `petal_length_cm`  | Petal length in centimetres            |
| `petal_width_cm`   | Petal width in centimetres             |

An 80/20 stratified split keeps class proportions stable across train and validation folds.

---

## Repository Layout

```
Classification/
├── README.md
├── artifacts/
│   └── .gitkeep
├── data/
│   └── iris.csv
├── demo.py
├── notebooks/
│   └── decision_tree_classification.ipynb
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

- `config.py` — centralises feature names, stopping criteria, file paths, and ensures directories exist.
- `data.py` — fetches the Iris dataset, renames columns to snake case, and exposes stratified train/validation splits.
- `pipeline.py` — trains the `DecisionTreeClassifier`, computes accuracy/precision/recall/F1 plus tree diagnostics, and persists artefacts with Joblib/JSON.
- `inference.py` — defines Pydantic request/response models, exposes a cached FastAPI service, and surfaces feature importances with every prediction.
- `train.py` — CLI entry point used in documentation, CI jobs, and manual retraining.
- `demo.py` — convenience script that emits a sample prediction from the command line.

---

## Extensions

1. **Pruning sweep** — iterate over `ccp_alpha` values to calibrate tree complexity and avoid overfitting.
2. **Class weights** — rebalance split criteria if you adapt the module to skewed datasets.
3. **Ensembles** — wrap the same preprocessing in `RandomForestClassifier` or `GradientBoostingClassifier` under the Ensemble Models roadmap.
4. **Explainability** — export the tree with `sklearn.tree.export_text` or render it with graphviz for presentation decks.
5. **Monitoring** — log feature importances and leaf indices in production to detect data drift.

---

## References

- L. Breiman, J. Friedman, R. Olshen, C. Stone. *Classification and Regression Trees.*
- Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. *An Introduction to Statistical Learning.*
- scikit-learn documentation: `sklearn.tree.DecisionTreeClassifier` and cost-complexity pruning.
