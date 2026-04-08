# Stacking Ensemble (Stacked Generalization)

## Overview

Stacking, also called Stacked Generalization, is an ensemble meta-algorithm that combines multiple base learners using a meta-learner. Instead of uniformly weighting all base models, stacking learns an optimal combination via a secondary model. This can discover non-intuitive, effective combinations.

---

## Algorithm Deep Dive

**Why it's called "stacking":** You stack predictions on top of each other as new features.

### Training Process (with k-fold cross-validation)

1. **Split training data into k folds**
2. **For each fold:**
   - Train base learners on k-1 folds
   - Generate predictions on hold-out fold
3. **Concatenate out-of-fold predictions:** These become features for meta-learner
4. **Train meta-learner on these meta-features**
5. **Retrain base learners on full training dataset** (for test-time predictions)

### Prediction Process

1. Base learners predict on new data
2. Their predictions form meta-features
3. Meta-learner predicts final output

### Why k-fold during training?

**Prevents data leakage:** If you trained base learners on full data then used their predictions on same data for meta-learner, you'd have information leakage. Meta-learner would memorize training set perfectly.

**Correct approach:** Out-of-fold predictions = base learners never saw these examples, so training meta-learner on OOF predictions is clean.

---

## Key Concepts

### **Base Learners**
- Use **diverse** models: e.g., Logistic Regression, Decision Tree, KNN
- Each should perform reasonably (>50% accuracy for binary classification)
- Errors should be **uncorrelated** - if base learners make same mistakes, stacking adds no value

### **Meta-Learner**
- Usually **simple:** Logistic Regression, Ridge Regression, or linear SVM
- Complex meta-learner risks overfitting (it's already seeing predictions from trained models)
- Typical choice: simple linear model on top of base prediction features

### **Complexity Trade-off**
- Training: k-fold × k base learners = k(k+1) model trainings (e.g., 5-fold = 30 trains)
- Improvement: typical 1-3% over best base learner (not always worth it)
- Rarely beats native multi-class implementations

---

## When to Use Stacking

Use when:
- Multiple diverse, well-tuned models available
- Want to squeeze out last 1-2% accuracy
- Computational budget allows k-fold training
- Base model errors are uncorrelated

Avoid when:
- Limited training time available (quadratic complexity)
- Base models have correlated errors
- Interpretability critical (stacking is a black box)
- Single strong model available (just use it)

---

## Example Scenario

Imagine you have three models:
- **Model A:** 90% accuracy but fails on certain patterns
- **Model B:** 88% accuracy, fails differently
- **Model C:** 85% accuracy, different weaknesses

Stacking learns: "When Model A and B disagree, trust B more. When all three agree, confidence is high."

This learned combination might achieve 91-92% by exploiting the complementary strengths.

---

## Practical Notes

### Diversity is Critical
Stacking only helps if base learners make different errors:
```python
prediction_diversity = correlation([base1_predictions, base2_predictions, ...])
# If correlation > 0.8, stacking may not help
# If correlation < 0.3, stacking can be powerful
```

### Meta-Learner Overfitting Risk
- Use simple meta-learner (linear model preferred)
- Consider regularization (L2 penalty)
- Monitor validation performance - stop if it worsens

### Don't Stack Everything
Not all models benefit from stacking:
- Two similar models (e.g., Logistic Regression + Ridge) → won't help
- Base models already at overfitting → stacking makes it worse
- Small dataset → not enough data for reliable meta-learner

---

## Files

- `demo.py` - Quick demo combining Logistic Regression, Decision Tree, KNN
- `src/pipeline.py` - Full k-fold stacking implementation
- `src/config.py` - Hyperparameters (cv_folds, base learners, meta-learner)
- `src/data.py` - Data loading (Iris dataset)
- `notebooks/` - Jupyter exploration and ablations

---

## Example Usage

```python
from src.pipeline import StackingEnsemblePipeline

pipeline = StackingEnsemblePipeline()
metrics = pipeline.train()
# metrics = {'accuracy': 0.965, 'precision_weighted': 0.965, ...}

# Prediction on new data
prediction = pipeline.base_learners['logistic_regression'].predict(new_X)  # Base predictions
meta_features = np.hstack([learner.predict_proba(new_X) for learner in pipeline.base_learners.values()])
final_prediction = pipeline.meta_learner.predict(meta_features)
```

---

## Performance Notes

- Rarely achieves >3% improvement over best base learner
- Improvement depends heavily on base learner diversity
- Correlated base models can actually decrease performance
- Meta-learner usually underwhelms expectations (often simple linear combinations work best)
- Consider before using: does 1-2% gain justify quadratic training time?

---

## Interview Tips

- Explain k-fold strategy and why it prevents leakage
- Mention diversity requirement (uncorrelated errors)
- State typical 1-3% improvement and trade-off with complexity
- Compare to voting (simpler, no training) and boosting (more powerful, sequential)
- Note: most production systems use Boosting (XGBoost) or Bagging (Random Forest) instead of Stacking

