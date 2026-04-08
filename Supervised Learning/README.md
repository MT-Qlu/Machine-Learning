# Supervised Learning

Complete, production-ready reference for supervised learning algorithms, ensemble techniques, and multi-class strategies. Designed for both learning and practical use: each module includes theory, implementation, experiments, and FastAPI deployment. This is a **flat structure with depth per topic** for easy navigation.

## What is Supervised Learning?

Supervised learning learns to predict a target variable given input features by learning from labeled examples. Divided into two tasks:

- **Classification:** Predict discrete class labels (e.g., Iris species, email spam/not spam). Output probabilities for each class.
- **Regression:** Predict continuous values (e.g., house price, stock return). Output single numeric value.

All modules here follow same workflow:
1. Load and explore data
2. Preprocess (scale, encode, handle missing values)
3. Train model on training set
4. Evaluate on test set (unseen during training)
5. Persist and deploy

## How to Navigate This Section

## Folder Structure

```
├── Linear Regression/              # Simple linear regression baseline
├── Logistic Regression/            # Binary/multi-class probabilistic classifier
├── Naive Bayes/                    # Fast probabilistic classifier
├── Regularized Models/             # Ridge, Lasso, Elastic Net
├── Generalized Linear Models/      # Poisson, Negative Binomial Regression
├── Decision Trees/                 # Tree-based model with interpretability
├── Support Vector Machine/         # SVM for classification & regression
├── K Nearest Neighbours/           # Distance-based non-parametric learning
├── Ensemble Models/                # Meta-algorithms combining multiple learners
│   ├── Bagging/                    # Bootstrap aggregating (Random Forest)
│   ├── Boosting/                   # Sequential error correction (AdaBoost, GBM, XGBoost)
│   ├── Stacking/                   # Meta-learner on base learner predictions
│   ├── Voting/                     # Simple majority vote or average
│   └── Meta-Algorithms/            # Research & advanced techniques
├── Probability Calibration/        # Calibration methods (Platt, Isotonic)
├── Multi-Class Strategies/         # Multi-class techniques
│   ├── One-vs-Rest/                # K binary classifiers (1 vs all)
│   └── One-vs-One/                 # K(K-1)/2 binary classifiers (pair-wise)
└── Time Series Forecasting/        # ARIMA, SARIMA, Prophet, Exponential Smoothing
```

---

## Quick Start

Each module follows the same structure:
- **`demo.py`** - Quick demonstration with sample predictions
- **`src/`** - Implementation (pipeline, config, data utilities)
- **`notebooks/`** - Jupyter explorations and experiments
- **`README.md`** - Detailed conceptual explanation
- **`data/`** & **`artifacts/`** - Datasets and trained models

**Run a demo:**
```bash
cd Linear\ Regression/
python demo.py
```

---

## Algorithm Reference

### Linear Regression
- **Core idea:** Fit a straight-line relationship between features and a continuous target by minimising mean squared error.
- **Use it when:** Relationships look roughly linear and interpretability matters; baseline every regression task with it.
- **Dial in:** Standardise features, check multicollinearity, and add regularisation (ridge/lasso) if coefficients explode.
- **Watch out:** Outliers and non-stationary data can dominate the fit - inspect residual plots and consider transformations.

## Logistic Regression
- **Core idea:** Estimate class probabilities via the sigmoid of a linear combination of features; interpret coefficients as log-odds.
- **Use it when:** You need a quick, explainable classifier for linearly separable problems or as a calibration baseline.
- **Dial in:** Scale inputs, balance classes (weights or sampling), and explore regularisation strength (`C` in scikit-learn).
- **Watch out:** High-dimensional sparse data can overfit without penalty terms - keep an eye on validation loss.

## Naive Bayes
- **Core idea:** Apply Bayes' theorem assuming feature independence, trading realism for speed and robustness on small data.
- **Use it when:** Working with text (bag-of-words), categorical features, or needing a surprisingly strong baseline.
- **Dial in:** Choose the right variant (Gaussian, Multinomial, Bernoulli) to match feature distributions.
- **Watch out:** Correlated features break the independence assumption; consider feature hashing or selection first.

## Support Vector Machines
### Classification (Breast Cancer)
- **Core idea:** Find the maximum-margin hyperplane separating classes, optionally projecting data into higher dimensions with kernels.
- **Use it when:** You have medium-sized datasets with clear margins or overlapping classes that benefit from kernel tricks.
- **Dial in:** Grid-search `C` and `gamma`, pick kernels (`linear`, `rbf`, `poly`) that reflect decision boundary complexity.
- **Watch out:** Scaling is mandatory; SVMs do not handle noisy, overlapping classes without careful tuning.

### Regression (California Housing)
- **Core idea:** Extend SVM to regression by fitting within an epsilon-insensitive tube, penalising only large deviations.
- **Use it when:** You need robust regression that ignores small errors and captures non-linear trends with kernels.
- **Dial in:** Adjust `C`, `epsilon`, and kernel parameters to balance flatness vs. sensitivity.
- **Watch out:** Performance degrades on large datasets due to quadratic complexity; sample or move to linear SVR when needed.

## K Nearest Neighbours (KNN)
### Classification
- **Core idea:** Predict the class most common among the `k` closest training examples using a distance metric.
- **Use it when:** Decision boundaries are irregular and you have a modest dataset with meaningful local structure.
- **Dial in:** Experiment with `k`, distance metrics (Euclidean, Manhattan), and weighting neighbours by inverse distance.
- **Watch out:** Scaling is essential; KNN is sensitive to irrelevant features and becomes slow with large datasets.

### Regression
- **Core idea:** Average the targets of the `k` nearest neighbours to estimate a continuous value.
- **Use it when:** Local patterns matter more than a global function and interpretability is less critical.
- **Dial in:** Tune `k`, distance weighting, and feature scaling; cross-validate to avoid over-smoothing or noisy predictions.
- **Watch out:** Outliers and heterogeneous feature scales distort neighbourhoods - normalise and consider outlier clipping.

## Decision Tree
### Classification (Iris)
- **Core idea:** Grow axis-aligned splits that maximise class purity and report feature importances for transparency.
- **Use it when:** You want an interpretable baseline on small-to-medium tabular data or need probability outputs without heavy tuning.
- **Dial in:** Adjust `max_depth`, `min_samples_leaf`, and `ccp_alpha` to balance bias/variance; inspect feature importances for sanity checks.
- **Watch out:** Deep trees memorise noise - validate with stratified splits and prune or cap depth when metrics diverge.

### Regression (California Housing)
- **Core idea:** Segment the feature space into regions with similar target means, predicting the average value per leaf.
- **Use it when:** You need quick non-linear baselines with explainable splits and ranked feature importance.
- **Dial in:** Tune depth/leaf thresholds and cost-complexity pruning; bucket geographic features when necessary.
- **Watch out:** Piecewise-constant predictions can jump at split boundaries; monitor residuals and revisit settings if variance spikes.

## Ensemble Models

Combining multiple weak learners into a strong composite model is one of machine learning's most powerful techniques. This section covers all major ensemble paradigms implemented in production-ready Python.

### Bagging / Random Forest
- **Core idea:** Bootstrap aggregating (Bagging) trains many models on random subsets of training data (sampling with replacement), then combines predictions via averaging (regression) or voting (classification). Random Forest specializes Bagging to decision trees, also randomizing feature subsets at splits.
- **Why it works:** Reduces variance by averaging uncorrelated predictions while maintaining low bias from each base learner. Success depends on base learner diversity and combining different parts of feature space.
- **Algorithm:** (1) Create B bootstrap samples by random sampling with replacement. (2) Train base model on each sample. (3) Average predictions (or vote). Out-of-bag score uses unseen bootstrap samples for free validation.
- **Hyperparameters:** n_estimators (more = better but diminishing returns after 100-500). max_samples (fraction of data per tree). max_features (randomness in splits). max_depth (base model complexity).
- **Use it when:** High-variance base learner (deep trees), need robust baseline, interpretable feature importance, computational budget allows parallel training.
- **Remember:** Set enough estimators; use out-of-bag score for validation without cross-validation overhead. Tune max_features for diversity (typically sqrt(features) for classification, features/3 for regression).
- **Interview tip:** Highlight robustness to missing data, natural feature importance metrics, parallel training potential, no hyperparameter tuning *required* to work reasonably well.
- **Watch out:** High bias if base learner too simple (e.g., shallow stumps). Correlated base models reduce ensemble benefit. Max_depth cap essential - uncapped trees hurt variance reduction.
\n### Boosting (AdaBoost, Gradient Boosting, Stochastic GBM, XGBoost)\n- **Core idea:** Sequentially train models where each focuses on examples mis-predicted by previous models. Combine via weighted sum, increasing weight for hard examples. Builds strong models from weak learners through iterative refinement.\n- **Why it works:** Reduces bias by focusing capacity on residual errors. Works best when base learner is weak but better than random. Less sensitive to feature scaling than Bagging.\n- **AdaBoost Algorithm:** (1) Initialize sample weights uniformly. (2) Train classifier on reweighted data. (3) Increase weight of misclassified examples. (4) Final prediction: weighted vote of all classifiers.\n- **Gradient Boosting:** (1) Initialize model (e.g., mean). (2) Fit shallow tree to residuals of previous model. (3) Add scaled tree to ensemble. (4) Repeat until convergence or n_estimators reached.\n- **Hyperparameters:** n_estimators (number of sequential models). learning_rate (shrinkage; lower = slower but often better). max_depth (base model depth, typically 3-5 for GBM). subsample (row sampling for stochasticity). colsample_bytree (column sampling for regularization).\n- **Use it when:** Bias-heavy problems (bagging underperforms). Large datasets with complex feature interactions. Can spend more training time. Need strong single model.\n- **Remember:** Learning rate vs. n_estimators trade-off: low learning rate + high n_estimators often beats high learning rate + few estimators (slower training, better generalization). Early stopping via validation set critical.\n- **Interview tip:** Explain sequential nature vs. Bagging parallelism. Mention regularization (shrinkage, subsampling, depth caps) essential for performance. XGBoost: sparse-aware splits, built-in regularization, handles missing data.\n- **Watch out:** Prone to overfitting without regularization. Requires careful tuning of learning rate and depth. Sensitive to feature scaling (not mandatory but helps). Training slower than Bagging - can't parallelize across iterations. Early stopping recommended: monitor validation loss, stop if no improvement.\n\n### Bagging vs. Boosting Summary\n\n| Aspect | Bagging | Boosting |\n|--------|---------|----------|\n| **Focus** | Reduces variance | Reduces bias |\n| **Training** | Parallel (independent samples) | Sequential (dependent iterations) |\n| **Base learner** | High-variance (e.g., deep trees) | Weak learner (e.g., stumps) |\n| **Sample reweighting** | Uniform (within bootstrap) | Adaptive (focus on hard examples) |\n| **Generalization** | Stable, less tuning needed | Powerful but needs regularization |\n| **Speed** | Faster training (parallel) | Slower (sequential) |\n| **When to use** | Quick baseline, robust predictions | Squeeze accuracy, complex interactions |\n

### Stacking
- **Core idea:** Train base learners on the dataset, then use their predictions as features for a meta-learner. Uses k-fold cross-validation on training data to generate out-of-fold predictions, avoiding data leakage.
- **Algorithm:** For each fold, train base learners on k-1 folds, predict on hold-out; concatenate to form meta-features. Retrain base learners on full training data. Meta-learner trained on meta-features, final predictions from base + meta.
- **Use it when:** Multiple diverse base learners available and you can tolerate additional training complexity. Typically improves over best single model by 1-3%.
- **Dial in:** Choose diverse base learners (e.g., Logistic Regression, Decision Tree, KNN); use simple meta-learner (Logistic Regression, Ridge). Tune k-fold splits to balance stability and training time.
- **Watch out:** Prone to overfitting if not using k-fold strategy. Meta-learner improvement is often modest. Requires careful base learner selection for diversity.

### Voting
- **Core idea:** Combine predictions from multiple trained classifiers via hard voting (majority class) or soft voting (average predicted probabilities).
- **Hard Voting:** Each classifier casts one vote; final prediction is the most common class. Simple, fully non-parametric.
- **Soft Voting:** Classifiers predict probability for each class; average probabilities and select highest. Works better with probability-calibrated models.
- **Use it when:** Multiple trained models available and you need a quick, interpretable ensemble. Ideal when models are already tuned - no retraining needed.
- **Dial in:** Mix diverse algorithms (linear, tree, distance-based) for best results. Assign optional weights to individual classifiers. Use soft voting if probabilities are well-calibrated.
- **Watch out:** Hard voting treats all models equally despite varying quality. Soft voting sensitive to poor probability calibration. Improvement modest unless base models diverse and uncorrelated.

## Probability Calibration
- **Core idea:** Post-process model predictions to ensure predicted probabilities match empirical frequencies.
- **Techniques:**
  - **Platt Scaling:** Fit sigmoid to model scores (works well for SVM, Tree models)
  - **Isotonic Regression:** Non-parametric monotonic fit (more flexible, needs more data)
- **Use it when:** Probabilistic outputs critical (e.g., medical diagnosis, pricing models).
- **Watch out:** Calibration reduces sharpness; only apply if probability accuracy is more important than confidence.

## Multi-Class Strategies

### One-vs-Rest (OvR)
- **Core idea:** For K classes, train K binary classifiers where each classifier distinguishes one class from all others. For prediction, run all K classifiers and choose the class with highest confidence.
- **Algorithm:** For class i, positive examples = class i, negative examples = everything else. Train binary classifier. At test time, generate scores from all K classifiers; predict argmax.
- **Complexity:** Training O(K x n) - linear in number of classes. Prediction O(K) - linear scaling.
- **Use it when:** Linear scalability with K matters (many classes). Native binary classifiers only available. Interpretability of one-vs-all boundaries important.
- **Dial in:** Works with any binary classifier (SVM, Logistic Regression, Decision Trees). Addresses class imbalance by reweighting if needed.
- **Watch out:** Class imbalance inherent (1 positive, K-1 negative per model) can bias decision boundaries. Adjacent classes might not separate cleanly.
- **Interview tip:** Simpler than OvO, scales better, often comparable accuracy. Native multi-class implementations (e.g., multi-class SVM) often preferred.

### One-vs-One (OvO)
- **Core idea:** For K classes, train K(K-1)/2 binary classifiers - one for each pair of classes. For prediction, all classifiers vote; final class is majority vote winner.
- **Algorithm:** For each pair (i, j), train binary classifier on examples from class i and j only. At test time, run all K(K-1)/2 classifiers; each votes for one class. Return class with most votes.
- **Complexity:** Training O(K2) classifiers but each on balanced binary subset. Prediction O(K2) - quadratic scaling.
- **Use it when:** Classes heavily imbalanced (OvO creates balanced sub-problems per pair). Want balanced training for each binary classifier. Can tolerate quadratic complexity.
- **Dial in:** Each pairwise problem is well-balanced regardless of global class distribution. Good for imbalanced multi-class problems.
- **Watch out:** K(K-1)/2 grows quadratically - becomes expensive for large K. Slower inference than OvR. Rarely beats native multi-class implementations despite added complexity.
- **Interview tip:** Mention when balanced pairwise problems matter (imbalanced data). Explain quadratic trade-off. Note that modern libraries implement this efficiently for certain algorithms (e.g., SVM).

## Time Series Forecasting
- **Shared setup:** All modules load the AirPassengers dataset, split chronologically, log MAE/RMSE/MAPE, and persist models with Joblib for FastAPI inference.

### ARIMA
- **Core idea:** Combine autoregression (AR), differencing (I), and moving averages (MA) to model stationary series.
- **Use it when:** After differencing removes trend/seasonality and residuals look white-noise.
- **Watch out:** Over-differencing kills signal; rely on ACF/PACF and diagnostics plots.

### SARIMA
- **Core idea:** Extend ARIMA with seasonal components to capture repeating yearly or monthly patterns.
- **Use it when:** Seasonality is obvious (e.g., monthly passengers) and needs explicit modelling.
- **Watch out:** Too many seasonal parameters explode training time - start small and inspect residuals.

### Prophet
- **Core idea:** Decompose time series into trend, seasonality, and holidays with an additive model that handles missing data gracefully.
- **Use it when:** You want fast, auto-tuned forecasts with interpretable components and built-in uncertainty intervals.
- **Watch out:** Default changepoint priors can underfit sudden regime shifts; loosen `changepoint_prior_scale` when trends jump.

### Exponential Smoothing (Holt-Winters)
- **Core idea:** Smooth level, trend, and seasonality exponentially, giving more weight to recent observations.
- **Use it when:** Seasonality is regular and you prefer a fast, classical baseline.
- **Watch out:** Incorrect seasonal period or damping choices lead to drift - double-check seasonal length and residual plots.

---
**How to use this file:** Skim before coding rounds, then open the specific module for hands-on pipelines, notebooks, and FastAPI services.
