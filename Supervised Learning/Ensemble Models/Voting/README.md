# Voting Ensemble

## Overview

Voting Ensemble combines predictions from multiple trained classifiers via:
- **Hard voting:** Take the most common prediction (majority vote)
- **Soft voting:** Average predicted probabilities, then predict the class with highest average

Simple, fast, and often effective - requires no additional training, just combining existing models.

---

## Algorithm Details

### Hard Voting (Plurality Voting)

Each classifier independently predicts a class; the class with most votes wins.

```
Model 1 predicts: Class A
Model 2 predicts: Class A  
Model 3 predicts: Class B
Final: Class A (2 votes > 1 vote)
```

**Advantages:**
- Fully non-parametric (no assumptions)
- Works with any classifier type
- Deterministic, easy to explain
- No tuning required

**Disadvantages:**
- Treats all models equally (no quality weighting)
- Ties possible with even number of models
- Ignores confidence/probability information

### Soft Voting (Average Probability)

Each classifier predicts **probability** for each class; average probabilities across classifiers, predict highest average.

```
Model 1: Class A (95%), Class B (5%)
Model 2: Class A (80%), Class B (20%)
Model 3: Class B (70%), Class A (30%)

Average:
Class A: (95 + 80 + 30) / 3 = 68.3%
Class B: (5 + 20 + 70) / 3 = 31.7%

Final: Class A (higher average probability)
```

**Advantages:**
- Uses full probability information
- Naturally weights confidence
- Better calibrated than hard voting
- Works with probability-outputting classifiers

**Disadvantages:**
- Requires calibrated probabilities
- If models output poor probabilities, voting amplifies errors
- Slightly more complex

---

## When to Use Hard vs. Soft Voting

| Scenario | Choose |
|----------|--------|
| Mix of classifiers (SVM, Trees, Linear models) | Hard voting |
| All models output well-calibrated probabilities | Soft voting |
| Speed critical | Hard voting (simpler) |
| Need confidence weighting | Soft voting |
| Uncertain about model quality | Hard voting (symmetric) |

---

## Key Concepts

### Diversity is Everything
Voting only beats base models if they **disagree on hard examples:**

**Bad case:** Two identical models
- If one fails, both fail → voting doesn't help
- No redundancy

**Good case:** Two uncorrelated models
- Model A: Strong on pattern X, weak on pattern Y
- Model B: Weak on X, strong on Y
- Voting leverages complementary strengths

**Measure diversity:**
```python
correlation_matrix = np.corrcoef([model1_predictions, model2_predictions, ...])
# Correlation < 0.7 = good diversity
# Correlation > 0.9 = too similar
```

### Weighting Models

Optional: assign different weights to different classifiers based on performance.

```python
VotingClassifier(
    estimators=[('model1', clf1), ('model2', clf2), ('model3', clf3)],
    voting='soft',
    weights=[3, 1, 2]  # Model 1 has 3x influence of Model 2
)
```

**When to use weights:**
- Models have vastly different accuracy
- Some models proven reliable on certain tasks
- Available computational budget to tune

**Caution:** Tuning weights on training set causes overfitting. Use cross-validation.

---

## When to Use Voting Ensemble

✅ **Use when:**
- Multiple trained models already available
- Need fast ensemble (no retraining)
- Computational budget limited
- Want interpretable combination (explicit voting)
- Base models are diverse

❌ **Avoid when:**
- Base models highly correlated (will vote same way)
- Only one strong model available (just use it)
- Need maximum accuracy (Boosting/Stacking exceed voting)
- Base model probabilities poorly calibrated (hard voting better)

---

## Practical Example

**Scenario:** Healthcare risk scoring for loan approval

Three models trained:
1. **Logistic Regression:** 85% accuracy, stable probabilities
2. **Gradient Boosting:** 88% accuracy, sometimes overconfident
3. **SVM with calibration:** 86% accuracy, well-calibrated

**Hard voting outcome:** Model opinion (87% accuracy). Fully transparent.

**Soft voting outcome:** Combines probability information (88% accuracy). Better captures confidence.

In practice: soft voting wins because different models trained on different data subsets reduce overfitting.

---

## Files

- `demo.py` - Quick demo comparing hard vs. soft voting on Iris
- `src/pipeline.py` - Voting implementation with metrics for both strategies
- `src/config.py` - Configuration (voting type, weights)
- `src/data.py` - Data loading (Iris dataset)
- `notebooks/` - Jupyter exploration and ablations

---

## Example Usage

```python
from src.pipeline import VotingEnsemblePipeline

pipeline = VotingEnsemblePipeline()
metrics = pipeline.train()
# Returns: {'hard_voting': {...}, 'soft_voting': {...}}

# Hard vote
prediction_hard = pipeline.hard_voting.predict(new_X)

# Soft vote
prediction_soft = pipeline.soft_voting.predict(new_X)

# Probabilities (soft voting only)
probabilities = pipeline.soft_voting.predict_proba(new_X)
```

---

## Performance Expectations

- **Hard voting:** Often matches or slightly exceeds best base model (1-2%)
- **Soft voting:** Can exceed hard voting if probabilities well-calibrated (1-3%)
- **Highly diverse models:** Improvement can reach 5% with soft voting
- **Correlated models:** Voting can actually decrease accuracy

---

## Comparison with Other Ensembles

| Ensemble | Training | Use Best For |
|----------|----------|--------------|
| **Voting** | No (models only) | Fast baseline, interpretability |
| **Bagging** | Yes (parallel) | Robustness, feature importance |
| **Boosting** | Yes (sequential) | Maximum accuracy gain |
| **Stacking** | Yes (complex) | Squeeze last 1-3% (rarely worth it) |

---

## Interview Tips

- Explain hard vs. soft voting difference
- Mention diversity requirement (uncorrelated predictions)
- State voting advantage: zero training cost, interpretable
- Explain soft voting sensitivity to probability calibration
- Compare to Bagging (parallel training) and Boosting (sequential accuracy)
- Note: voting often a good "last check" before submitting ensemble model

