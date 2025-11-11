# Benchmark Tools

**Location:** `Machine-Learning/Essentials Toolkit/Benchmark Tools`

This toolkit provides a lightweight harness for benchmarking the models that live across the repository. It focuses on reproducible experiment definitions (YAML/JSON), dependable metric coverage, and automated report generation that can be shared with the evaluation and monitoring stacks.

## Directory Layout

- `benchmark_tools/` – Python package that powers dataset loading, model instantiation, metric evaluation, and reporting.
- `configs/` – Ready-to-run configuration examples for common regression and classification suites.
- `output/` – Default drop-zone for generated CSV summaries and JSON detail reports (kept empty by `.gitkeep`).
- `cli.py` – Convenience entry point so you can launch experiments with `python "Essentials Toolkit/Benchmark Tools/cli.py" --config <file>`.

## Quick Start

1. Install the project dependencies (see `requirements.txt`).
2. Pick or author a configuration file under `configs/`.
3. Run the CLI:

	 ```bash
	 cd Machine-Learning
	 python "Essentials Toolkit/Benchmark Tools/cli.py" --config "Essentials Toolkit/Benchmark Tools/configs/iris_classification.yaml"
	 ```

4. Inspect the printed table for a high-level summary and open the timestamped files under `Essentials Toolkit/Benchmark Tools/output/` for the persisted artefacts.

## Configuration Schema

All scenarios conform to the schema implemented in `benchmark_tools.config`:

```yaml
name: unique-experiment-id
task_type: classification | regression
description: short free-form summary

	loader: alias | dotted.path | /absolute/path/to/data.csv
	target: optional-target-column (required when loading tabular files)
	features: [optional, list, of, feature, columns]
	options: {}  # extra kwargs forwarded to the loader
models:
	- name: Friendly name in reports
		estimator: alias | dotted.path.to.Estimator
		parameters: {scikit-learn style kwargs}
metrics:
	- name: registered-metric-name
		alias: optional-short-name-in-reports
		prediction: labels | probabilities | decision
		average: macro | micro | weighted  # for multi-class metrics
		greater_is_better: true | false  # optional override
split:
	strategy: train_test | kfold | stratified_kfold
	test_size: 0.2
	folds: 5
	shuffle: true
	random_state: 42
output:
	directory: Essentials Toolkit/Benchmark Tools/output/custom
	save_csv: true
	save_json: true
	timestamped: true
metadata:
	owner: you
```

> **Tip:** Any field can be omitted if you are happy with the defaults captured by the Pydantic models.

## Supported Aliases

### Dataset loaders

The toolkit ships with shortcuts for common scikit-learn datasets:

- `iris`
- `wine`
- `breast_cancer`
- `digits`
- `california_housing`
- `diabetes`

You can reference any other dataset via a dotted path (for example `sklearn.datasets:fetch_openml`) or by pointing to a local CSV/Parquet file.

### Model factories

Aliases map to scikit-learn estimators (and can be extended easily):

- `logistic_regression`
- `random_forest_classifier`
- `random_forest_regressor`
- `gradient_boosting_classifier`
- `gradient_boosting_regressor`
- `elastic_net`, `lasso`, `ridge`, `linear_regression`
- `svc`, `svr`
- `knn_classifier`, `knn_regressor`
- `decision_tree_classifier`, `decision_tree_regressor`
- `ada_boost_classifier`, `ada_boost_regressor`
- `sgd_classifier`, `sgd_regressor`

To use a custom estimator supply the dotted import path (e.g. `fastapi_app.models.pipeline:CustomEstimator`).

### Metric registry

`benchmark_tools.metrics` merges:

- The canonical definitions from `Essentials Toolkit/metrics/metrics.py` (MAE, RMSE, MSE, MAPE, sMAPE, MASE, R², Huber, Log-Cosh, Quantile, binary/categorical cross entropy, hinge).
- Scikit-learn’s metrics (accuracy, precision, recall, F1, ROC AUC, average precision, log loss, explained variance, median/mean absolute error, MSE, RMSE, R², and more).

Each metric specifies whether it expects label predictions, probabilities, or decision scores so the runner can request `predict`, `predict_proba`, or `decision_function` as needed.

## Outputs

Every run produces:

- A tabular summary (DataFrame / CSV) with the mean and standard deviation for each metric across folds.
- A JSON document capturing fold-level scores and the raw metadata from your configuration.

You can toggle persistence or point to a different directory through the `output` block.

## Extending the Toolkit

- **New metrics:** register them in `benchmark_tools/metrics.py` by adding to `_SKLEARN_REGISTRY` or wrapping functions from third-party libraries.
- **Custom datasets:** implement a loader function that returns a scikit-learn style `Bunch`/tuple and expose it through a dotted path.
- **Additional model aliases:** extend `MODEL_ALIASES` inside `benchmark_tools/models.py`.

Contributions and refinements are welcome—mirror the patterns already used by the sample configs and keep documentation inline so downstream users can bootstrap quickly.
