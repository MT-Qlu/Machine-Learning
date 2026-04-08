# Ensemble Learning Suite

**Location:** `Machine-Learning/Supervised Learning/Ensemble Models`

Ensemble methods combine multiple base learners to deliver stronger generalisation than any individual model. This suite mirrors the repository’s standard project structure (data, src, notebooks, artifacts, demo) so you can train, evaluate, and serve bagging and boosting pipelines alongside other supervised modules.

---

## Why ensembles work

- **Bagging** reduces variance by averaging many high-variance models (e.g., trees).
- **Boosting** reduces bias by sequentially correcting residual errors.
- **Diversity** across learners improves robustness and generalisation.

## When to use

- Tabular datasets with non-linear feature interactions.
- You need strong baselines with minimal feature engineering.
- Model stability and performance matter more than interpretability.

## Comparison: When to Use Each Paradigm

| Method | Best For | Training | Complexity | Typical Gain |
|--------|----------|----------|-----------|--------------|
| **Bagging** | High-variance models, robustness | Parallel | Low | 2-5% |
| **Boosting** | Accuracy push, bias reduction | Sequential | Medium-High | 5-15% |
| **Stacking** | Expert combination | Sequential | Very High | 1-3% |
| **Voting** | Already-trained models | None | Trivial | 2-5% |

**Decision guide:**
- **Quick accuracy:** Use Boosting (XGBoost/GBM)
- **Robustness:** Use Bagging (Random Forest)
- **Models already trained:** Use Voting
- **Many uncorrelated experts:** Use Stacking (if gain > 1%)
- **Need interpretability:** Use Bagging or Voting
- **Limited training time:** Use Voting, then Bagging

---

## Practical Workflow

1. Train strong individual models.
2. Compare Voting ensemble - does combining beat the best?
3. If bias is bottleneck: use Boosting.
4. If variance is bottleneck: use Bagging.
5. Rarely: use Stacking only if 1-3% gain justifies complexity.

---

## Structure

- **Bagging/** - Bootstrap aggregating and Random Forest implementations.
- **Boosting/** - Gradient boosting, stochastic GBM, AdaBoost, and XGBoost variants.
- **Stacking/** - Meta-learner on base learner predictions with cross-validation.
- **Voting/** - Hard and soft voting classifiers.
- **Meta-Algorithms/** - Research techniques (Cascade, Blending, Mixture of Experts).

Each submodule includes its own README with dataset details, CLI commands, and FastAPI integration tips.
