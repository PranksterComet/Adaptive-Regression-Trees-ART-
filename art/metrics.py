"""Error metrics for model and tree evaluation."""

from __future__ import annotations

import numpy as np


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after flattening.")
    return arr


def _validate_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_vector(y_true, "y_true")
    y_pred = _as_vector(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must match, got {y_true.shape} and {y_pred.shape}.")
    return y_true, y_pred


def sum_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    residual = y_true - y_pred
    return float(np.dot(residual, residual))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("Cannot compute mean squared error on an empty array.")
    return sum_squared_error(y_true, y_pred) / y_true.size


def relative_l2_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 1e-12,
) -> float:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    denom = max(float(np.linalg.norm(y_true)), float(floor))
    return float(np.linalg.norm(y_true - y_pred) / denom)


def pointwise_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 1e-12,
) -> np.ndarray:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    return np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), float(floor))


def median_pointwise_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 1e-12,
) -> float:
    return float(np.median(pointwise_relative_error(y_true, y_pred, floor=floor)))


def max_pointwise_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 1e-12,
) -> float:
    return float(np.max(pointwise_relative_error(y_true, y_pred, floor=floor)))
