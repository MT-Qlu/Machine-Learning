# Meta-Algorithms (Advanced Research & Blending Ensemble)

## Overview

Advanced ensemble and meta-learning techniques beyond standard supervised learning. Combines multiple base learners using sophisticated meta-learning strategies. Currently implements **Blending** with extensible architecture for additional meta-algorithms.

## Current Implementation: Blending Ensemble

**What it does:** Trains independent base learners on training data, generates predictions on validation set, then trains meta-learner to combine those predictions optimally.

**Key features:**
- Uses holdout set (not k-fold CV like Stacking)
- Simpler and faster than Stacking
- Less data required for meta-learner
- Better when computational budget limited

**Base learners tested:**
- Random Forest (ensemble baseline)
- Gradient Boosting (sequential learning)
- Decision Tree (interpretability)

**Algorithm:**
1. Split data: train (60%), validation/blending set (20%), test (20%)
2. Train base learners on train set
3. Generate meta-features: predictions on blending set
4. Train meta-learner (Logistic Regression) on meta-features
5. Evaluate ensemble on test set

**Performance:**
- Blending accuracy on Iris: typically 96-98%
- Individual base learner accuracies: 80-95%
- Demonstrates meta-learner advantage over base models

## Structure

- `src/pipeline.py` - BlendingPipeline core: train/predict/save/load
- `src/config.py` - Configuration: base learner params, paths, random states
- `src/data.py` - Iris loading and preprocessing utilities
- `src/inference.py` - Pydantic models and MetaAlgorithmService for FastAPI deployment
- `src/train.py` - Command-line training entry point
- `demo.py` - Quick demonstration script
- `notebooks/` - Exploratory analysis (voting analysis, base learner comparison)
- `data/` - Raw/processed iris data
- `artifacts/` - Trained models, scaler, metrics JSON

## Potential Extensions

### Cascade Ensemble
- Sequential classifiers where each level focuses on harder examples filtered by previous level
- Use when: Dataset is large and ranking/filtering by confidence is valuable
- Implementation: Sequence of classifiers with confidence threshold gating

### Mixture of Experts (MoE)
- Gating network learns which specialist (expert) model to use per sample
- Use when: Different regions of feature space need different models
- Implementation: Learnable gating function with softmax over experts

### Feature-weighted Linear Stacking (FWLS)
- Learns per-feature importance weights for combining base learner predictions
- Use when: Feature-space complexity varies per dimension
- Implementation: Ridge regression on feature-weighted meta-features

### Snapshot Ensembling
- Extract multiple model snapshots from single training trajectory
- Use when: Computational budget is limited
- Implementation: Save model periodically during SGD/cyclic learning rate

### Diversity-Aware Ensemble
- Weight base learners by diversity metrics (disagreement, margin, entropy)
- Use when: Need principled way to combine diverse classifiers
- Implementation: Quadratic programming for optimal diversity weights

## Usage

Train meta-algorithms:

```python
python demo.py
```

Train with custom config:

```python
from src.pipeline import BlendingPipeline
from src.config import MetaAlgorithmsConfig

config = MetaAlgorithmsConfig(
    blending_holdout_ratio=0.3,
    n_estimators_rf=200,
    meta_learner_type='logistic'
)

pipeline = BlendingPipeline(config)
metrics = pipeline.train()
```

Inference:

```python
from src.inference import get_service, MetaAlgorithmRequest

service = get_service()

request = MetaAlgorithmRequest(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)

response = service.predict(request)
print(f"Predicted: {response.predicted_class_name}")
print(f"Confidence: {max(response.confidence_scores):.2%}")
```

## Comparison with Stacking

| Aspect | Stacking | Blending |
|--------|----------|----------|
| **Meta-feature generation** | k-fold CV (multiple folds) | Single holdout set |
| **Training time** | Slower (k times) | Faster (single pass) |
| **Data efficiency** | Uses full dataset for meta-features | Wastes ~20% on holdout |
| **Overfitting risk** | Lower (diverse folds) | Higher (fixed holdout) |
| **Implementation complexity** | More complex | Simpler |
| **When to use** | When accuracy critical | When speed critical |

## Notebooks

- `voting_analysis.ipynb` - Compares hard vs soft voting with blending ensemble
- `base_learner_comparison.ipynb` - Individual accuracy vs blending gain
- `meta_feature_analysis.ipynb` - Visualization of meta-feature space and meta-learner decision boundary

## Contributing

To add a new meta-algorithm:

1. Create new `<AlgorithmName>Pipeline` class in `pipeline.py`
2. Inherit configuration in `config.py` if needed
3. Add Pydantic request/response models to `inference.py`
4. Add service factory to `inference.py`
5. Create new entry point in `train.py` or separate module
6. Add demo to `demo.py` or create `demo_<name>.py`
