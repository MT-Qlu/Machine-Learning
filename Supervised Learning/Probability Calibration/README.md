# Probability Calibration

## Overview

**Calibration** makes a classifier's predicted probabilities reliable. When a well-calibrated model says an event has 70% probability, that event occurs roughly 70% of the time. Without calibration, predictions might be overconfident (100%) or underconfident (50%) even with good ROC-AUC.

---

## Why Calibration Matters

### Real-World Example

**Scenario:** Loan approval system

- **Uncalibrated Model A:** Says 95% chance of loan repayment
  - In reality: Only 60% actually repay
  - **Problem:** Bank sets risk threshold assuming 95%, suffers losses

- **Calibrated Model B:** Says 60% chance of loan repayment (after calibration)
  - In reality: Roughly 60% actually repay
  - **Advantage:** Bank can confidently threshold at P ≥ 75% to limit defaults

Both models might have identical ROC-AUC (ranking quality), but Model B is **trustworthy** for probability-based decisions.

### When Calibration Critical

**Medical diagnosis:** Must report reliable probability to physicians
**Insurance pricing:** Premiums based on predicted risk
**Fraud detection:** Thresholds set by business rules on probability
**Portfolio optimization:** Expected value calculations require calibrated probabilities
**Fairness audits:** Calibration across demographic groups essential

**When less critical:** Ranking/recommendation (only relative order matters, not absolute probability)

---

## How Calibration Works

### The Problem

Raw classifier outputs (logits, distances, raw probabilities) often don't match observed frequencies:

```
Model confidence | Observed frequency
10%              | 15% (underconfident)
50%              | 55% (roughly correct)
90%              | 75% (overconfident)
```

### The Solution

Fit a secondary model on a **held-out calibration set**:

1. Get raw scores/probabilities from primary trained model
2. Observe true labels on separate calibration data
3. Learn mapping: raw_score → calibrated_probability
4. Apply mapping at inference time

**Key:** Use separate data for calibration (never on training set - causes overfitting).

---

## Two Main Calibration Techniques

### Platt Scaling

**Idea:** Fit sigmoid function to transform scores to probabilities

$$
P_{cal}(Y=1 | X) = \frac{1}{1 + e^{-(aS + b)}}
$$

where $S$ is the raw classifier score, and $(a, b)$ are learned parameters.

**Advantages:**
- Simple (only 2 parameters)
- Fast training and inference
- Works well when: raw scores roughly follow sigmoid shape
- Good for: SVM, tree classifiers, linear models

**Disadvantages:**
- Assumes sigmoid shape (not flexible)
- Can fail if raw scores have non-monotonic shape

**Best for:** SVM (raw scores not probabilities by default), tree-based ensembles

### Isotonic Regression

**Idea:** Learn arbitrary monotonic mapping (step-wise non-parametric)

Fits a piecewise constant function that's monotonically increasing:

```
Raw Score | Calibrated Probability
0.0-0.2   → 0.05
0.2-0.4   → 0.12
0.4-0.6   → 0.55
0.6-0.8   → 0.75
0.8-1.0   → 0.92
```

**Advantages:**
- Flexible (no assumption on shape)
- Often better calibration than Platt
- Works even with non-monotonic raw outputs

**Disadvantages:**
- Many parameters (needs more calibration data)
- Risk of overfitting on small calibration sets
- Slower than Platt (lookup table or interpolation)

**Best for:** When you have enough calibration data and suspect non-sigmoid relationship

---

## Comparison: Platt vs. Isotonic

| Aspect | Platt Scaling | Isotonic Regression |
|--------|---------------|-------------------|
| **Flexibility** | Low (sigmoid only) | High (arbitrary shape) |
| **Data needed** | ~100 examples | ~500+ examples |
| **Training time** | < 1s | ~1s |
| **Inference time** | Very fast | Fast |
| **When to use** | Limited calibration data | Abundant calibration data |
| **Risk** | Underfitting | Overfitting |

---

## Calibration Metrics

### Expected Calibration Error (ECE)
Average absolute difference between predicted confidence and observed frequency across bins:

$$
ECE = \sum_{i=1}^{N_{bins}} \frac{|\mathcal{B}_i|}{N} |acc(\mathcal{B}_i) - conf(\mathcal{B}_i)|
$$

**Lower is better.** Range: [0, 1]. Well-calibrated: ECE < 0.05.

### Brier Score
Mean squared error of probabilities:

$$
Brier = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

**Lower is better.** Well-calibrated model: Brier matches baseline entropy.

### Reliability Diagram
Visual: plot predicted probability vs. empirical frequency for bins.
- Perfect calibration: diagonal line
- Above diagonal: overconfident
- Below diagonal: underconfident

---

## Practical Workflow

### Step 1: Train Primary Classifier
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
```

### Step 2: Calibrate

**Option A: Hold-out Calibration Set**
```python
from sklearn.calibration import CalibratedClassifierCV

cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
cal_clf.fit(X_calibrate, y_calibrate)  # Separate calibration data

# Predict with calibrated probabilities
p_calibrated = cal_clf.predict_proba(X_test)
```

**Option B: Cross-Validation (no separate data needed)**
```python
cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
cal_clf.fit(X_train, y_train)  # Uses CV internally
```

### Step 3: Evaluate
```python
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Brier score
brier = brier_score_loss(y_test, cal_clf.predict_proba(X_test)[:, 1])

# Calibration curve (for visualization)
prob_true, prob_pred = calibration_curve(y_test, cal_clf.predict_proba(X_test)[:, 1])
```

---

## Common Pitfalls

### Problem: Calibrating on Training Data
Leads to overfitting - calibration layer memorizes training set.
Solution: **Always use held-out calibration set.**

### Problem: Collecting Wrong Base Model
If primary classifier has high bias/variance, calibration can't fix it.
Solution: **Calibrate good (not mediocre) classifiers.**

### Problem: Using Too Much Calibration Data
Using 50% of data for calibration wastes training potential.
Solution: **Use 10-20% for calibration, rest for training.**

### Problem: Trusting Calibration Beyond Original Domain
Calibration learned on one data distribution may not transfer.
Solution: **Recalibrate on new domain if performance drifts.**

---

## When NOT to Calibrate

- **Ranking task:** Only relative order matters (use raw scores)
- **Already calibrated:** Neural nets with softmax + cross-entropy often well-calibrated
- **Computational budget:** Calibration adds inference latency (minor but real)
- **Interpretability:** Calibration layer is a black box - harder to explain

---

## Files & Implementation

- **Platt Scaling/** - Sigmoid calibration on top of logistic regression
  - `demo.py`, `src/pipeline.py`, `src/data.py`
  - Shows ECE before/after calibration

- **Isotonic Regression/** - Flexible non-parametric calibration
  - `demo.py`, `src/pipeline.py`, `src/data.py`
  - Comparison with Platt on different data distributions

---

## Example Output

```json
{
  "baseline_model": {
    "accuracy": 0.88,
    "brier_score": 0.15,
    "ece": 0.12
  },
  "platt_scaled": {
    "accuracy": 0.88,  // unchanged
    "brier_score": 0.11,  // improved!
    "ece": 0.05  // greatly improved
  },
  "isotonic_calibrated": {
    "accuracy": 0.88,
    "brier_score": 0.10,  // best
    "ece": 0.03  // best
  }
}
```

Note: Accuracy unchanged (calibration only adjusts probability), but probability quality (Brier, ECE) greatly improved.

---

## Interview Tips

- **Explain:** Calibration makes probabilities reliable, not just accurate class predictions
- **Difference:** ROC-AUC measures ranking; calibration measures probability truthfulness
- **Techniques:** Platt (2 params, simple) vs. Isotonic (flexible, needs data)
- **When:** Critical for probability-based decisions (insurance, medical), less so for ranking
- **Pitfall:** Never calibrate on training data - causes overfitting
- **Metrics:** ECE (error), Brier (MSE), reliability diagrams (visual)
- **Cost:** Minimal - just a post-processing layer

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

- `Platt Scaling/` - Fits a sigmoid on top of raw scores.
- `Isotonic Regression/` - Learns a monotonic step-wise mapping for flexible calibration.

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
