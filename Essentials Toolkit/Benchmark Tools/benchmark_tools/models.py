"""Model factory and alias registry."""

from __future__ import annotations

import importlib
from typing import Callable, Dict

from sklearn import ensemble, linear_model, neighbors, svm, tree
from sklearn.base import BaseEstimator

from .config import ModelConfig

MODEL_ALIASES: Dict[str, Callable[..., BaseEstimator]] = {
    "logistic_regression": linear_model.LogisticRegression,
    "ridge": linear_model.Ridge,
    "lasso": linear_model.Lasso,
    "elastic_net": linear_model.ElasticNet,
    "linear_regression": linear_model.LinearRegression,
    "random_forest_classifier": ensemble.RandomForestClassifier,
    "random_forest_regressor": ensemble.RandomForestRegressor,
    "gradient_boosting_classifier": ensemble.GradientBoostingClassifier,
    "gradient_boosting_regressor": ensemble.GradientBoostingRegressor,
    "ada_boost_classifier": ensemble.AdaBoostClassifier,
    "ada_boost_regressor": ensemble.AdaBoostRegressor,
    "sgd_classifier": linear_model.SGDClassifier,
    "sgd_regressor": linear_model.SGDRegressor,
    "svc": svm.SVC,
    "svr": svm.SVR,
    "knn_classifier": neighbors.KNeighborsClassifier,
    "knn_regressor": neighbors.KNeighborsRegressor,
    "decision_tree_classifier": tree.DecisionTreeClassifier,
    "decision_tree_regressor": tree.DecisionTreeRegressor,
}


def instantiate_model(config: ModelConfig) -> BaseEstimator:
    """Instantiate a model according to the provided configuration."""

    estimator_key = config.estimator
    if estimator_key in MODEL_ALIASES:
        factory = MODEL_ALIASES[estimator_key]
    else:
        if ":" in estimator_key:
            module_name, attr = estimator_key.split(":", 1)
        else:
            module_name, attr = estimator_key.rsplit(".", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)

    instance = factory(**config.parameters)

    if not isinstance(instance, BaseEstimator):
        raise TypeError(f"Instantiated object is not a scikit-learn estimator: {estimator_key}")

    return instance
