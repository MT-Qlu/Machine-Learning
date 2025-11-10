<!-- markdownlint-disable MD013 -->
# Prophet Forecasting — AirPassengers

This module packages Facebook Prophet (now simply Prophet) for the AirPassengers dataset. It mirrors the repository’s production-ready layout: configuration, scripted training, persisted artefacts, FastAPI integration, notebook exploration, and a CLI demo. Use it to experiment with additive or multiplicative seasonality, changepoint control, and probabilistic forecast intervals.

---

## Learning Objectives

- Fit a Prophet model with multiplicative yearly seasonality to airline passenger data.
- Evaluate forecasts against a held-out slice and persist metrics.
- Return median forecasts plus lower/upper bounds through the shared FastAPI registry.
- Extend the baseline with custom holidays, extra regressors, or hyperparameter sweeps.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/Prophet/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "mae": 10.8,
     "rmse": 14.9,
     "mape": 4.9
   }
   ```

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast**

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/prophet_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

   Prophet returns median forecasts (`yhat`) and uncertainty intervals (`yhat_lower`, `yhat_upper`).

5. **Open the notebook**

   `notebooks/prophet_forecasting.ipynb` mirrors the scripted pipeline, including trend/seasonality plots and residual checks.

---

## Mathematical Foundations

Prophet decomposes a time series into trend, seasonality, and holiday components:

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t.
$$

- $g(t)$ models the trend via piecewise linear or logistic growth with changepoints.
- $s(t)$ captures seasonal patterns using Fourier series.
- $h(t)$ encodes user-defined holidays or events.

The model is fit using Stan, providing posterior uncertainty estimates that translate into forecast intervals.

---

## Dataset

- **Source**: AirPassengers (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv` (`ds`, `y` columns for Prophet).
- **Frequency**: Monthly (`MS`).

---

## Repository Layout

```
Prophet/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── prophet_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── pipeline.py
│   └── train.py
└── artifacts/
    └── .gitkeep
```

---

## Extensions

1. **Holiday calendars** — inject known travel peaks to improve accuracy.
2. **Scenario analysis** — adjust growth rate priors or cap/floor for capacity planning.
3. **Cross-validation** — use Prophet’s built-in `cross_validation` utilities for rolling evaluation.

---

## References

- Sean J. Taylor, Benjamin Letham. "Forecasting at scale." *The American Statistician*, 2018.
- Prophet documentation: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
