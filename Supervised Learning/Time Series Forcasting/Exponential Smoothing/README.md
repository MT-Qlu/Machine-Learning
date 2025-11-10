<!-- markdownlint-disable MD013 -->
# Exponential Smoothing — AirPassengers

This module implements Holt-Winters Exponential Smoothing for the AirPassengers dataset. It adheres to the repository’s production-style structure: configuration, scripted training, persisted artefacts, FastAPI integration, notebook exploration, and a CLI demo. Use it as a lightweight seasonal baseline alongside ARIMA, SARIMA, and Prophet.

---

## Learning Objectives

- Fit an additive-trend, multiplicative-seasonal Holt-Winters model.
- Evaluate forecasts against a chronological holdout slice.
- Deploy forecasts through the shared FastAPI registry.
- Compare smoothing-based performance against ARIMA-family and Prophet models.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/Exponential Smoothing/src/train.py"
   ```

   Example output:

   ```json
   {
     "mae": 14.6,
     "rmse": 18.9,
     "mape": 6.8
   }
   ```

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast**

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/exponential_smoothing_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

5. **Open the notebook**

   `notebooks/exponential_smoothing.ipynb` mirrors the training pipeline and visualises forecast overlays.

---

## Mathematical Foundations

Holt-Winters smoothing maintains level ($L_t$), trend ($T_t$), and seasonal ($S_t$) components:

$$
\begin{aligned}
L_t &= \alpha \frac{y_t}{S_{t-s}} + (1 - \alpha)(L_{t-1} + T_{t-1}), \\
T_t &= \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}, \\
S_t &= \gamma \frac{y_t}{L_t} + (1 - \gamma) S_{t-s},
\end{aligned}
$$

with forecasts given by

$$
\hat{y}_{t+h} = (L_t + h T_t) S_{t-s+h}.
$$

This additive trend + multiplicative seasonality configuration handles exponential growth while remaining computationally lightweight.

---

## Dataset

- **Source**: AirPassengers (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv`.
- **Seasonality**: 12-month cycle.

---

## Repository Layout

```
Exponential Smoothing/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── exponential_smoothing.ipynb
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

1. **Component tuning** — experiment with additive vs. multiplicative configurations.
2. **Damped trend** — enable damping for longer forecast horizons.
3. **Hybrid ensembles** — average forecasts from ARIMA/SARIMA/Prophet for robustness.

---

## References

- Charles C. Holt, "Forecasting Trends and Seasonals by Exponentially Weighted Moving Averages", 1957.
- Peter R. Winters, "Forecasting Sales by Exponentially Weighted Moving Averages", 1960.
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
