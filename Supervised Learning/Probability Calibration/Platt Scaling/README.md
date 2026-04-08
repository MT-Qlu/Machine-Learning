# Platt Scaling — Sigmoid Calibration

**Location:** `Machine-Learning/Supervised Learning/Calibration/Platt Scaling`

Applies a logistic/sigmoid calibration layer to raw classifier scores, improving probability estimates for downstream decision making.

---

## Theory in brief

Platt scaling fits a sigmoid on top of uncalibrated scores $s(x)$:

$$
\hat{p}(y=1\mid x) = \sigma(a\,s(x) + b) = \frac{1}{1 + e^{-(a s(x) + b)}}
$$

The parameters $(a, b)$ are learned by minimising log-loss on a calibration split. This works well when the miscalibration is **approximately sigmoid-shaped**.

## When to use

- Binary classifiers that output scores or margins (SVMs, boosted models).
- Small to medium datasets where a parametric calibration curve is preferable to avoid overfitting.
- Situations where you want a simple, monotonic correction.

## Beginner example

Suppose a credit-risk model outputs a raw score of 2.1 (not a probability). Platt scaling learns a sigmoid that maps score 2.1 to, say, 0.74 probability of default. That single curve makes the model’s outputs usable for business decisions like “approve if risk < 0.2.”

## Highlights

- Synthetic binary dataset with intentional class imbalance.
- Baseline logistic regression followed by scikit-learn `CalibratedClassifierCV` (`method="sigmoid"`).
- Metrics include Brier score loss and expected calibration error (ECE).
- FastAPI service exposing calibrated probabilities for integration tests.

## Workflow

1. Train the base classifier on the training split.
2. Fit the sigmoid calibration on held-out data (or use cross-validation).
3. Compare Brier score and reliability curves before/after calibration.
4. Persist calibrated artefacts and expose them through FastAPI.
