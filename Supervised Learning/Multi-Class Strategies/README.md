# Multi-Class Strategies

## Overview

Many binary classifiers need adaptation for multi-class (K > 2) problems. This section covers two fundamental strategies for extending binary classifiers to K-class scenarios. Understanding these trade-offs is essential for both algorithm selection and debugging multi-class problems.

---

## Binary vs. Multi-class Problem

### Binary Classification
Output: 2 classes (e.g., "spam" or "not spam")
Natural for binary classifiers (Logistic Regression, SVM)

### Multi-class Classification
Output: K classes where K > 2 (e.g., Iris: 3 classes)
Requires strategy to extend binary classifiers

**Native multi-class:** Some algorithms handle K-class directly (Decision Trees, Neural Networks)

**Binary adapters:** Use binary classifiers as building blocks (One-vs-Rest, One-vs-One)

---

## Strategy Comparison

### One-vs-Rest (OvR)

**Idea:** For each class i, train a binary classifier: "class i" vs. "everything else"

**Process:**
- K classes → K binary classifiers
- Classifier 1: Class 0 vs. {1, 2, ..., K-1}
- Classifier 2: Class 1 vs. {0, 2, ..., K-1}
- ...
- Classifier K: Class K-1 vs. {0, 1, ..., K-2}

**Prediction:** Confidence from all K classifiers; argmax wins

**Complexity:**
- Training: O(K x dataset_size) - linear in classes
- Prediction: O(K × binary_prediction_time)

**Example:** 10-class problem requires 10 binary classifiers

### One-vs-One (OvO)

**Idea:** For every pair of classes (i, j), train binary classifier: "class i" vs. "class j"

**Process:**
- K classes → K(K-1)/2 binary classifiers
- Classifiers for pairs: (0,1), (0,2), ..., (0,K-1), (1,2), ..., (K-1 vs all previous)

**Prediction:** All classifiers vote; most votes wins

**Complexity:**
- Training: O(K^2 x smaller_dataset_size) - quadratic in classes
- Prediction: O(K² × binary_prediction_time)

- Example: 10-class problem requires 45 binary classifiers (10×9/2)

---

## Detailed Comparison

| Aspect | One-vs-Rest | One-vs-One |
|--------|-------------|-----------|
| **# of Classifiers** | K | K(K-1)/2 |
| **Training Set Per Classifier** | Full (1 vs. K-1) | Balanced pair (i vs. j) |
| **Class Distribution** | Heavy imbalance (1 pos, K-1 neg) | Balanced per pair |
| **Training Complexity** | O(K × n) Linear | O(K² × n/2) Quadratic |
| **Prediction Complexity** | O(K) Linear | O(K²) Quadratic |
| **Storage** | K models | K(K-1)/2 models |
| **Typical Accuracy** | Good | Comparable or slightly better |
| **Interpretability** | Clear (1-vs-all boundaries) | Pairwise decisions harder to interpret |
| **When Better** | Balanced data, speed | Imbalanced data, small K |

---

## When to Use Each

### Use One-vs-Rest When:

**Many classes:** K = 50+. OvO becomes K(K-1)/2 approximately 1250 models - impractical.

**Classes roughly balanced:** OvR's class imbalance within each binary problem is unavoidable but manageable.

**Speed critical:** Linear scaling vs. quadratic.

**Interpretability needed:** "Class 3 vs. all others" easier to explain than pairwise votes.

**Computational constraints:** Fewer models to train and store.

### Use One-vs-One When:

**Severe class imbalance:** Each pair is naturally balanced, avoiding OvR's imbalance problem.

**Small number of classes:** K = 2-10. OvO still manageable (10-45 models).

**Accuracy over speed:** Can spend extra training and prediction time.

✅ **Binary classifier doesn't scale to imbalanced data:** e.g., SVM often performs better OvO for imbalanced problems.

✅ **Coverage needed:** Sometimes O vO catches patterns OvR misses due to balanced sub-problems.

---

## Practical Scenarios

### Email Classification (7 Topics)

| Scenario | Choice | Why |
|----------|--------|-----|
| Real-time, balanced classes | **OvR** | 7 models, linear, fast inference |
| High accuracy needed, offline | **OvO** | 21 models, balanced training, can afford quadratic |

### Medical Diagnosis (15 Diseases, Imbalanced)

| Scenario | Choice | Why |
|----------|--------|-----|
| Severe class imbalance | **OvO** | Each pair balanced, SVM performance improves |
| Production latency <100ms | **OvR** | Linear prediction, manageable |

---

## Files

- **One-vs-Rest/** - OvR implementation with SVM using Iris dataset
  - `demo.py`, `src/pipeline.py`, READMEs
  - Compares OvR vs. native multi-class SVM

- **One-vs-One/** - OvO implementation with SVM using Iris dataset
  - `demo.py`, `src/pipeline.py`, READMEs
  - Shows pairwise voting mechanism

---

## Reference Implementation

Both strategies available via scikit-learn:

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

# OvR
ovr = OneVsRestClassifier(SVC())
ovr.fit(X_train, y_train)

# OvO
ovo = OneVsOneClassifier(SVC())
ovo.fit(X_train, y_train)

# Compare
prediction_ovr = ovr.predict(X_test)
prediction_ovo = ovo.predict(X_test)
```

---

## Performance Expectations

**Binary Classifier Baseline:** 85% accuracy (example)

| Strategy | K=3 | K=5 | K=10 | K=50 |
|----------|-----|-----|------|------|
| **OvR** | 84-86% | 81-84% | 75-80% | 60-70% |
| **OvO** | 84-86% | 82-84% | 76-80% | N/A (impractical) |
| **Native (if available)** | 86-88% | 84-86% | 80-82% | 70-75% |

*Note: Native multi-class implementations usually outperform adapters because they optimize K-class objective directly.*

---

## Interview Tips

- **Explain both strategies clearly** with examples
- **Know complexity:** OvR linear, OvO quadratic in K
- **Trade-off:** Speed vs. balanced sub-problems
- **Mention:** Native multi-class usually better (optimization over K, not K binary problems)
- **Imbalance handling:** OvO naturally handles via balanced pairs
- **When to use:** OvR for many classes, OvO for severe imbalance or small K
- **Comparison:** Stacking or ensemble could combine both strategies

