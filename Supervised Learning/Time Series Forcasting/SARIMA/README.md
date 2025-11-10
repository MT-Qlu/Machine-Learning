<!-- markdownlint-disable MD013 -->
# SARIMA Forecasting — AirPassengers

This module extends the ARIMA workflow with explicit seasonality using a Seasonal ARIMA (SARIMA) model. It follows the repository’s production-style template: reproducible configuration, scripted training, persisted artefacts, FastAPI integration, an exploratory notebook, and a CLI demo. Use it to capture seasonal effects in univariate series and deploy the forecasts through the shared API.

---

## Learning Objectives

- Understand how SARIMA augments ARIMA with seasonal autoregressive and moving-average terms.
- Train, evaluate, and persist a SARIMA$(1,1,1)\times(1,1,1)_{12}$ model on the AirPassengers dataset.
- Compare seasonal vs. non-seasonal performance using held-out metrics.
- Serve seasonal forecasts via the unified FastAPI registry.
- Extend the baseline with seasonal diagnostics, grid searches, or exogenous regressors.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/SARIMA/src/train.py"
   ```

   Example output:

   ```json
   {
     "mae": 12.9,
     "rmse": 16.5,
     "mape": 6.0
   }
   ```

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast**

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/sarima_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

5. **Explore the notebook**

   Open `notebooks/sarima_forecasting.ipynb` for ACF/PACF diagnostics, training parity checks, and visual comparisons.

---

## Mathematical Foundations

SARIMA introduces seasonal differencing and seasonal AR/MA components:

$$
\Phi_p(L^s) (1 - L^s)^D \phi_p(L) (1 - L)^d X_t = \Theta_q(L^s) \theta_q(L) \varepsilon_t,
$$

where $L$ is the lag operator and $s$ is the seasonal period (12 for monthly data). This structure captures repeating annual patterns while still modelling non-seasonal dynamics. The $(1,1,1)\times(1,1,1)_{12}$ configuration provides a strong baseline for AirPassengers.

---

## Dataset

- **Source**: AirPassengers (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv`.
- **Seasonality**: 12-month period.

---

## Repository Layout

```
SARIMA/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── sarima_forecasting.ipynb
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

1. **Seasonal grid search** — tune $(P, D, Q)$ with AIC/BIC.
2. **Holiday regressors** — incorporate exogenous dummy variables for known peaks.
3. **Rolling origin** — compute walk-forward validation to assess deployment stability.

---

## References

- George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung. *Time Series Analysis: Forecasting and Control*.
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
- statsmodels documentation: `statsmodels.tsa.statespace.sarimax.SARIMAX`.
