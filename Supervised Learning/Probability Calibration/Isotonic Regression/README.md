# Isotonic Regression — Non-parametric Calibration

**Location:** `Machine-Learning/Supervised Learning/Calibration/Isotonic Regression`

Isotonic regression fits a monotonic calibration curve to raw classifier scores, making it ideal when probabilities need flexible adjustment beyond a sigmoid.

---

## Theory in brief

Isotonic calibration learns a **monotonic** function $f$ that maps scores to probabilities by solving:

$$
\min_{f \in \mathcal{M}} \sum_{i=1}^{n} \big(y_i - f(s_i)\big)^2, \quad \text{subject to } f \text{ non-decreasing}
$$

This produces a piecewise-constant calibration curve that can correct complex miscalibration patterns.

## When to use

- You see non-sigmoid miscalibration (overconfidence at some ranges, underconfidence at others).
- Larger datasets where a flexible calibration curve won’t overfit.
- You need a monotonic correction without assuming a parametric form.

## Beginner example

Imagine a medical model that is too confident for low-risk patients but under-confident for high-risk patients. Isotonic regression learns a flexible, step-like curve that pushes low probabilities down and high probabilities up while staying monotonic. The result is a probability scale that matches reality across the full range.

## Highlights

- Same synthetic dataset as Platt scaling for baseline comparison.
- scikit-learn `CalibratedClassifierCV` with `method="isotonic"`.
- Calibration diagnostics reported via reliability diagrams and ECE.
- FastAPI-ready inference service returning calibrated probabilities.

## Workflow

1. Train a base classifier and obtain raw scores.
2. Fit isotonic regression on a held-out calibration set.
3. Plot reliability curves and compute ECE/Brier.
4. Persist calibrated artefacts for API inference.
