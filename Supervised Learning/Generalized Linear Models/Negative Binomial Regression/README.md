# Negative Binomial Regression — Over-dispersed Counts

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models/Negative Binomial Regression`

Use this module when count data exhibit variance larger than the mean, violating Poisson assumptions. It leverages `statsmodels` to fit a Negative Binomial GLM with a log link.

---

## Theory in brief

Negative Binomial regression extends Poisson by adding a dispersion parameter $\alpha$:

$$
\mathbb{E}[y] = \mu, \qquad \mathrm{Var}(y) = \mu + \alpha \mu^2
$$

with the same log link for the mean:

$$
\log(\mu) = X\beta + \log(\text{exposure}).
$$

## When to use

- Count data with **over-dispersion** (variance >> mean).
- Incident or demand series with bursty behaviour.
- Scenarios where Poisson residuals show heavy tails.

## Highlights

- Synthetic call-center dataset with over-dispersed ticket counts.
- Train/evaluate workflow built on pandas + statsmodels.
- Metrics include pseudo R² and mean absolute error.
- FastAPI-compatible inference wrapper for easy deployment.

## Diagnostics & tips

- Inspect dispersion estimates; if $\alpha \to 0$, Poisson may suffice.
- Compare deviance and AIC across Poisson vs Negative Binomial.
- Use rate offsets when exposure varies by sample.
