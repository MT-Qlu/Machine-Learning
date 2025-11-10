"""Reusable optimisation routines for custom training loops."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

State = Dict[str, np.ndarray]


def gradient_descent(params: np.ndarray, grads: np.ndarray, *, lr: float = 1e-2) -> np.ndarray:
    """One step of vanilla gradient descent."""

    return params - lr * grads


def sgd(params: np.ndarray, grads: np.ndarray, *, lr: float = 1e-2) -> np.ndarray:
    """Alias for gradient descent when gradients come from a single sample."""

    return gradient_descent(params, grads, lr=lr)


def mini_batch(params: np.ndarray, grads: np.ndarray, *, lr: float = 1e-2) -> np.ndarray:
    """Mini-batch gradient descent uses the same update rule as GD."""

    return gradient_descent(params, grads, lr=lr)


def momentum(
    params: np.ndarray,
    grads: np.ndarray,
    *,
    lr: float = 1e-2,
    momentum_coeff: float = 0.9,
    state: State | None = None,
) -> Tuple[np.ndarray, State]:
    """Momentum-based gradient descent."""

    velocity = state.get("velocity") if state else np.zeros_like(params)
    velocity = momentum_coeff * velocity - lr * grads
    params = params + velocity
    return params, {"velocity": velocity}


def nesterov(
    params: np.ndarray,
    grads: np.ndarray,
    *,
    lr: float = 1e-2,
    momentum_coeff: float = 0.9,
    state: State | None = None,
) -> Tuple[np.ndarray, State]:
    """Nesterov accelerated gradient."""

    velocity = state.get("velocity") if state else np.zeros_like(params)
    velocity = momentum_coeff * velocity - lr * grads
    params = params + (momentum_coeff * velocity) - lr * grads
    return params, {"velocity": velocity}


def adagrad(
    params: np.ndarray,
    grads: np.ndarray,
    *,
    lr: float = 1e-2,
    epsilon: float = 1e-8,
    state: State | None = None,
) -> Tuple[np.ndarray, State]:
    """Adaptive gradient algorithm."""

    history = state.get("history") if state else np.zeros_like(params)
    history += np.square(grads)
    adjusted_lr = lr / (np.sqrt(history) + epsilon)
    params = params - adjusted_lr * grads
    return params, {"history": history}


def rmsprop(
    params: np.ndarray,
    grads: np.ndarray,
    *,
    lr: float = 1e-3,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    state: State | None = None,
) -> Tuple[np.ndarray, State]:
    """RMSProp optimiser."""

    history = state.get("history") if state else np.zeros_like(params)
    history = beta * history + (1 - beta) * np.square(grads)
    adjusted_lr = lr / (np.sqrt(history) + epsilon)
    params = params - adjusted_lr * grads
    return params, {"history": history}


def adam(
    params: np.ndarray,
    grads: np.ndarray,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    state: State | None = None,
    step: int | None = None,
) -> Tuple[np.ndarray, State]:
    """Adam optimiser with bias correction."""

    if state is None:
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
    else:
        m = state["m"]
        v = state["v"]
        t = state.get("t", 0)

    t = t + 1 if step is None else step
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * np.square(grads)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return params, {"m": m, "v": v, "t": t}


__all__ = [
    "gradient_descent",
    "sgd",
    "mini_batch",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adam",
]
