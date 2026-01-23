# Probability Calibration

**Location:** `Machine-Learning/Supervised Learning/Calibration`

Reliable probabilities are critical for thresholding, risk scoring, and cost-sensitive decisions. This suite demonstrates two popular calibration techniques built atop a logistic regression backbone.

## When to use calibration

- **Decision thresholds matter**: fraud, medical triage, or churn where the cost of false positives/negatives differs.
- **Ranking is fine but probabilities are not**: ROC-AUC is strong, but predicted probabilities are systematically high/low.
- **Downstream cost models**: expected value decisions require well-calibrated probabilities, not just class labels.

## Core idea

Calibration learns a mapping from raw classifier scores $s(x)$ to calibrated probabilities $\hat{p}(x)$ so that

$$
P(Y=1 \mid \hat{p}(X)=p) \approx p.
$$

In practice, you fit a **calibration layer** on a held-out calibration split or via cross-validation.

## Beginner example

Imagine a spam filter that says an email has a “90% chance” of being spam, but in reality only 70% of those emails are spam. Calibration learns a correction curve so that when the model says 0.9, the observed frequency is closer to 0.9. After calibration, you can set thresholds like “block if probability ≥ 0.8” with much more confidence.

- `Platt Scaling/` — Fits a sigmoid on top of raw scores.
- `Isotonic Regression/` — Learns a monotonic step-wise mapping for flexible calibration.

Each module preserves the repo’s standard structure with datasets, src code, notebooks, artifacts, and demos.

## Evaluation signals

- **Brier score** (lower is better): mean squared error of probabilistic predictions.
- **Expected Calibration Error (ECE)**: gap between predicted confidence and observed frequency.
- **Reliability diagrams**: visual summary of calibration quality across probability bins.

## Suggested workflow

1. Train a baseline classifier (logistic regression in this suite).
2. Split off a calibration set or use `CalibratedClassifierCV`.
3. Compare raw vs calibrated probabilities using ECE and Brier.
4. Persist calibrated artefacts for FastAPI inference.
