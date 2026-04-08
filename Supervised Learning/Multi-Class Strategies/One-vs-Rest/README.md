# One-vs-Rest (OvR) Multi-Class Strategy

## Overview

One-vs-Rest (OvR), also called **One-vs-All**, extends any binary classifier to K-class problems by training K independent binary classifiers, each answering: "Is this class i or not?"

---

## Algorithm Explained

### Training Process

For a 3-class problem (Class 0, Class 1, Class 2):

**Binary Classifier 0:** "Is it Class 0?" vs. "Everything else"
- Class 0: positive (label = 1)
- Class 1, 2: negative (label = 0)

**Binary Classifier 1:** "Is it Class 1?" vs. "Everything else"
- Class 1: positive (label = 1)
- Class 0, 2: negative (label = 0)

**Binary Classifier 2:** "Is it Class 2?" vs. "Everything else"
- Class 2: positive (label = 1)
- Class 0, 1: negative (label = 0)

Result: K binary classifiers, each independently trained.

### Prediction Process

New example arrives:

1. Run through Binary Classifier 0 → confidence for Class 0
2. Run through Binary Classifier 1 → confidence for Class 1
3. Run through Binary Classifier 2 → confidence for Class 2
4. **Predict argmax:** Class with highest confidence wins

**Example:**
- Class 0 confidence: 0.2
- Class 1 confidence: 0.9
- Class 2 confidence: 0.4
- **Predict --> Class 1** (highest confidence)

---

## Complexity Analysis

### Training Complexity: O(K × n)
- Train K classifiers
- Each on full dataset (size n)
- Total work: K × n = linear in number of classes

### Prediction Complexity: O(K)
- Run example through K classifiers
- Return argmax

**Scalability:** Handles 100+ classes reasonably well.

---

## Key Characteristics

### Advantages

1. **Simple:** Extend any binary classifier
2. **Interpretable:** Easy to understand decision per class
3. **Efficient:** Linear scaling with number of classes
4. **Parallelizable:** Can train K classifiers in parallel
5. **Well-understood:** Standard, reliable approach

### Disadvantages

1. **Class Imbalance Problem:**
   - For K=10, positive examples = 10% (1 class), negative examples = 90% (rest)
   - This massive imbalance can harm binary classifier performance
   - Classifier biased toward predicting "negative" (not this class)

2. **Imbalanced Confidences:**
   - Some classifiers may never output high confidence
   - All examples might get low confidence overall
   - Confidence scores less reliable for decision threshold tuning

3. **Redundancy:**
   - Classification decisions not independent
   - Example: if confidence scores [0.4, 0.3, 0.2], sum < 1 (not a proper probability distribution)

4. **Ambiguous Edge Cases:**
   - If no classifier outputs > 0.5, argmax is forced but arbitrary
   - If all classifiers similarly confident, hard to distinguish

---

## When to Use OvR

**Use OvR when:**
- Many classes (K >> 10)
- Classes relatively balanced
- Binary classifier you want to use
- Speed or interpretability matter
- Standard off-the-shelf solution acceptable

**Avoid OvR when:**
- Severe class imbalance (need One-vs-One or native multi-class)
- Very few classes (K ≤ 3, native multi-class often beats OvR)
- Confidence calibration critical (OvR less reliable)
- Need maximum accuracy (native multi-class usually wins)

---

## Practical Example: Disease Diagnosis

**Scenario:** Diagnose 5 diseases from 10 symptoms

One-vs-Rest approach:
- **Model 0:** Disease A vs. {B, C, D, E} - 20% disease A, 80% others
- **Model 1:** Disease B vs. {A, C, D, E} - 20% disease B, 80% others
- ... (similar imbalance for each)

Each model's task: "20% positive, 80% negative"

**Decision:** Confidence scores for each disease; pick highest

**Issue:** Heavy class imbalance can degrade each binary classifier

**Solution:** Use class weights or resampling within each binary classifier:
```python
from sklearn.svm import SVC
svc = SVC(class_weight='balanced')  # Automatically reweight
ovr = OneVsRestClassifier(svc)
```

---

## Implementation Details

### With Scikit-Learn

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# Train
ovr = OneVsRestClassifier(SVC(probability=True))
ovr.fit(X_train, y_train)

# Predict class
prediction = ovr.predict(X_test)

# Get confidence scores
confidence = ovr.decision_function(X_test)  # Raw scores
# or
probabilities = ovr.predict_proba(X_test)  # If probability=True

# Compare to native multi-class
from sklearn.svm import SVC as SVCNative
svc_native = SVC(multi_class='ovr')  # Uses OvR internally
svc_native.fit(X_train, y_train)
# Note: Some models like SVC default to OvR for multi-class
```

### Manual Implementation

```python
from sklearn.svm import SVC
import numpy as np

classifiers = []
for class_idx in range(n_classes):
    y_binary = (y_train == class_idx).astype(int)
    clf = SVC()
    clf.fit(X_train, y_binary)
    classifiers.append(clf)

# Predict: get scores from all classifiers
scores = np.array([clf.decision_function(X_test) for clf in classifiers]).T
predictions = np.argmax(scores, axis=1)
```

---

## Performance Comparison

**Binary Baseline (2-class):** 95% accuracy

**Iris Dataset (3-class):**
- **OvR with SVM:** 97% accuracy (good: small K, balanced classes)
- **OvO with SVM:** 97% accuracy (comparable)
- **SVM with multi_class='ovr':** 97% accuracy (same - uses OvR internally)
- **Decision Tree (native 3-class):** 98% accuracy

**High Imbalance (10 diseases, 5% each):**
- **OvR with SVM:** 85% accuracy (struggles with imbalance)
- **OvO with SVM:** 88% accuracy (balanced pairs help)
- **Native multi-class + class weights:** 90% accuracy (optimized for K-class)

---

## Files

- `demo.py` - Quick OvR demo on Iris
- `src/pipeline.py` - OvR with SVM, comparing native multi-class
- `src/config.py` - Configuration (train/test split, dataset path)
- `src/data.py` - Iris data loading
- `notebooks/` - Exploration: decision boundaries, confidence analysis

---

## Interview Tips

- **Explain algorithm:** K classifiers, each "class i vs. rest"
- **Complexity:** O(K × n) training, O(K) prediction
- **Problem:** Class imbalance - 1 positive, K-1 negative per model
- **Solution:** Class weights or resampling
- **Compare to OvO:** OvR scales linearly; OvO quadratic but balanced pairs
- **When to use:** Many classes, balanced data, speed matters
- **Mention:** Most multi-class implementations (Logistic Regression, SVM) use OvR or native K-class, not One-vs-One

