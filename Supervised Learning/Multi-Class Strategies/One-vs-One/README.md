# One-vs-One (OvO) Multi-Class Strategy

## Overview

One-vs-One (OvO) extends binary classifiers to K-class problems by training K(K-1)/2 binary classifiers for every **pair** of classes. Each classifier learns to distinguish Class i from Class j. Prediction via majority voting.

---

## Algorithm Explained

### Training Process

For a 3-class problem (Class 0, Class 1, Class 2):

**Binary Classifier (0,1):** Class 0 vs. Class 1
- Training data: only examples from Class 0 and Class 1
- Naturally balanced: 50% Class 0, 50% Class 1

**Binary Classifier (0,2):** Class 0 vs. Class 2
- Training data: only examples from Class 0 and Class 2
- Naturally balanced: 50% Class 0, 50% Class 2

**Binary Classifier (1,2):** Class 1 vs. Class 2
- Training data: only examples from Class 1 and Class 2
- Naturally balanced: 50% Class 1, 50% Class 2

Result: 3 binary classifiers, each trained on balanced 2-class data.

### Prediction Process

New example arrives; run through all classifiers:

1. Classifier (0,1) → predicts 0 or 1
2. Classifier (0,2) → predicts 0 or 2
3. Classifier (1,2) → predicts 1 or 2

**Tally votes:**
- Class 0: got votes from (0,1) and (0,2) → 2 votes
- Class 1: got vote from (0,1) and (1,2) → 2 votes
- Class 2: got vote from (1,2) → 1 vote

**Predict --> Class 0 or 1** (tied at 2 votes; tiebreaker picks first)

---

## Complexity Analysis

### Training Complexity: O(K²)
- K(K-1)/2 classifiers (quadratic in K)
- Each on balanced 2-class subset ≈ 2n/K examples
- Total: K(K-1)/2 × 2n/K ≈ O(n × K)
- **Quadratic in K:** K=10 → 45 classifiers. K=50 → 1225!

### Prediction Complexity: O(K²)
- Run through K(K-1)/2 = O(K²) classifiers
- Tally votes via majority

**Scalability:** Becomes expensive for many classes (K=50+ impractical).

---

## Key Advantages vs. One-vs-Rest

### Naturally Balanced Sub-problems
- Each pair is 50/50 by design
- No artificial class imbalance like OvR ("1 class vs 99% others")
- Some algorithms (SVM, Logistic Regression) perform better on balanced data

### Handles Global Imbalance Better
Example: K=10 classes, but Class A is 50% of data, Class B-J are 5% each:
- OvR: Class A doesn't struggle (40% vs 60%), but Class D (5% vs 95%) faces severe imbalance
- OvO: Class A vs. B (roughly 50/50), Class D vs. E (10/10 or similar), all balanced

This makes OvO more robust to imbalanced datasets.

### Transparent Pairwise Decisions
Can inspect: "Classifiers agree: Class A vs B (clear), but confused on A vs C"

---

## Key Disadvantages vs. One-vs-Rest

### Quadratic Explosion
- 10 classes -> 45 classifiers
- 50 classes -> 1225 classifiers (storage and inference nightmare)
- Scales poorly compared to OvR's linear K

### Slow Inference
- Prediction requires K(K-1)/2 runs through classifiers
- For real-time/low-latency systems, prohibitive
- Linear vs. quadratic matters for large K

### Complex Model Management
- Many more models to train, monitor, version, deploy
- Storage and API serving more complicated

---

## When to Use OvO

**Use when:**
- Severe class imbalance (pairwise balancing helps)
- K is at most 10 (manageable number of classifiers)
- Accuracy is more important than speed (can trade inference latency)
- Binary classifier doesn't scale to imbalance (e.g., SVM)
- Offline or batch prediction (no real-time constraint)

**Avoid when:**
- K is greater than 20 (quadratic explosion)
- Real-time prediction required
- Memory limited
- Balanced data already (OvR sufficient, simpler)
- Native multi-class available (usually better)

---

## Comparison with OvR

| Aspect | One-vs-Rest | One-vs-One |
|--------|-------------|-----------|
| Train Complexity | O(K × n) Linear | O(K² × n) Quadratic |
| Predict Complexity | O(K) Linear | O(K²) Quadratic |
| Sub-problem Imbalance | Severe (1 vs K-1) | None (always 50/50) |
| Handles Data Imbalance | Struggles | Handles well |
| Storage | K models | K(K-1)/2 models |
| Scalability | 100+ classes OK | 20 classes max |
| Typical Accuracy (balanced) | Good | Comparable |
| Typical Accuracy (imbalanced) | Poor | Better |
| Implementation Complexity | Simple | Moderate |

---

## Practical Example: Hospital Diagnosis (10 Diseases)

**Global imbalance (collected patient data):**
- Disease A (heart attack): 40%
- Disease B (stroke): 25%
- Diseases C-J (rare): 1-4% each

**One-vs-Rest Problems:**
- Model for A (vs. rest): 40% vs. 60% (OK)
- Model for D (vs. rest): 2% vs. 98% (severe imbalance!)
- Issue: Rare disease models struggle to learn pattern

**One-vs-One Problems:**
- Model for (A, B): 40/(40+25) = 62% A, 38% B (manageable imbalance)
- Model for (D, E): 2/(2+2) = 50/50 (balanced!)
- Model for (A, D): 40/(40+2) = 95/5 (still imbalanced but different training dynamics)

**Result:** OvO typically outperforms OvR on severely imbalanced medical data.

---

## Implementation (Scikit-Learn)

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

# Train
ovo = OneVsOneClassifier(SVC(probability=True))
ovo.fit(X_train, y_train)

# Predict
prediction = ovo.predict(X_test)

# Number of classifiers
print(f"Classifiers trained: {len(ovo.estimators_)}")  # K(K-1)/2
```

---

## Performance Expectations

**Example: Imbalanced 10-class problem**

| Method | Accuracy | Macro Recall (all classes) | Training Time |
|--------|----------|---------------------------|---------------|
| **One-vs-Rest** | 79% | 65% (rare classes weak) | 12s |
| **One-vs-One** | 85% | 80% (balanced subproblems) | 25s |
| **Native Multi-class + Weights** | 87% | 83% (optimized for K-class) | 8s |

OvO wins on accuracy for imbalanced; native wins overall.

---

## Files

- `demo.py` - OvO demo on Iris; shows pairwise voting
- `src/pipeline.py` - OvO with SVM, voting mechanism
- `src/config.py` - Configuration
- `src/data.py` - Iris loading
- `notebooks/` - Exploration: voting analysis, class pair performance

---

## Interview Tips

- **Explain:** K(K-1)/2 classifiers, one per pair of classes
- **Key benefit:** Naturally balanced sub-problems (50/50 each)
- **Key cost:** Quadratic complexity (K² classifiers and predictions)
- **When:** Severe imbalance, few classes, offline prediction
- **When not:** Many classes, real-time, or balanced data
- **Vs. OvR:** Better for imbalanced; OvR more scalable
- **Vs. native:** Native multi-class usually faster and simpler; OvO only preferred for specific imbalance scenarios

