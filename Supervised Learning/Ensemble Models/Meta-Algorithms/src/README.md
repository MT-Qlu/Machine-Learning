"""README for Meta-Algorithms source code."""

# Meta-Algorithms Source Code

This directory contains the implementation of advanced ensemble and meta-learning techniques.

## Current Implementation

### Blending Ensemble (`pipeline.py`, `inference.py`, `train.py`)
- Meta-algorithm that trains multiple base learners, then a meta-learner on holdout predictions
- Simpler alternative to stacking using fixed holdout rather than k-fold CV
- Faster training, lower data requirements for meta-learner
- Suitable when computational budget is limited

## Core Modules

- `config.py` - Configuration management for meta-algorithms
- `data.py` - Iris dataset loading and preprocessing
- `pipeline.py` - BlendingPipeline core implementation with train/predict
- `inference.py` - Pydantic models and service class for deployment
- `train.py` - Command-line training entry point
- `__init__.py` - Package initialization
