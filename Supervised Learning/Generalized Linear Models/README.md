# Generalised Linear Models

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models`

This suite introduces count-based regression techniques where Gaussian assumptions break down. Each module adheres to the repo’s standard structure, enabling quick swaps between GLMs and existing baselines.

---

## GLM refresher

Generalised Linear Models extend linear regression by allowing **non-Gaussian** response distributions and **link functions**:

$$
g(\mu) = X\beta, \qquad y \sim \text{Exponential Family}
$$

For count data, the canonical link is **log**, which ensures non-negative predictions.

## When to use

- **Counts** (events per time, tickets per day, incident rates).
- **Skewed outcomes** where variance grows with the mean.
- **Exposure-aware modelling** (offsets for time-at-risk or population size).

## Beginner example

Imagine predicting daily website incidents. Linear regression might produce negative numbers, which make no sense. A GLM with a log link keeps predictions positive and models how incidents scale with traffic, making the outputs realistic.

## Modules

- `Poisson Regression/` — Predict event counts with a log link and Poisson likelihood (scikit-learn implementation).
- `Negative Binomial Regression/` — Handle over-dispersed counts via `statsmodels` GLM utilities.

Use these when dealing with traffic, sales, or incident counts where variance grows with the mean.

## Workflow guidance

1. Inspect the mean–variance relationship; if variance rises with mean, prefer GLMs over OLS.
2. Start with Poisson; if over-dispersion appears, move to Negative Binomial.
3. Evaluate using deviance, MAE, and calibration of predicted rates.
4. Persist artefacts and expose FastAPI endpoints for consistent deployment.
