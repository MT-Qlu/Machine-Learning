# Poisson Regression — Count Modelling

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models/Poisson Regression`

This module predicts event counts (e.g., support tickets per day) using a log-link Poisson regression. It demonstrates how to handle non-negative integer targets where variance scales with the mean.

---

## Theory in brief

Poisson regression models counts with a log link:

$$
y \sim \text{Poisson}(\mu), \qquad \log(\mu) = X\beta + \log(\text{exposure})
$$

The optional **exposure offset** lets you model rates (events per unit time/population).

## When to use

- Count outcomes with non-negative integers.
- Mean approximately equals variance (or as a starting point before testing over-dispersion).
- You need rate modelling with exposure or time-at-risk.

## Beginner example

Suppose you track the number of customer support tickets per day. A Poisson model learns how tickets increase when usage rises, and it never predicts negative counts. If you add an exposure offset (like number of active users), the model predicts tickets per user instead of raw counts.

## Features

- Synthetic dataset capturing baseline events, exposure, and promotions.
- scikit-learn `PoissonRegressor` pipeline with feature scaling.
- Evaluation metrics tailored to count data (Poisson deviance, mean absolute error).
- FastAPI-compatible inference service to plug into production workflows.

## Diagnostics & tips

- **Over-dispersion**: if variance >> mean, consider Negative Binomial.
- **Zero inflation**: many zeros may require zero-inflated models.
- **Deviance residuals**: inspect for systematic bias or misfit.
