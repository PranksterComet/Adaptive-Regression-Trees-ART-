"""Objective functions for split optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import RegressionModel, WeightedRegressionModel


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class SoftObjectiveResult:
    loss: float
    grad: np.ndarray
    left_model: RegressionModel
    right_model: RegressionModel
    pi: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftObliqueRidgeObjective:
    """Soft sigmoid-gated objective for weighted-fit-compatible models."""

    temperature: float
    model_template: WeightedRegressionModel
    weight_floor: float = 1e-12

    def value_and_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> SoftObjectiveResult:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        if theta.shape[0] != X.shape[1] + 1:
            raise ValueError(f"theta must have length {X.shape[1] + 1}, got {theta.shape[0]}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not hasattr(self.model_template, "fit_weighted"):
            raise TypeError("model_template must provide fit_weighted(X, y, weights).")

        w = theta[:-1]
        z = float(theta[-1])

        scores = (X @ w - z) / self.temperature
        pi = sigmoid(scores)

        left_model = self.model_template.clone()
        right_model = self.model_template.clone()
        left_model.fit_weighted(X, y, weights=1.0 - pi, weight_floor=self.weight_floor)
        right_model.fit_weighted(X, y, weights=pi, weight_floor=self.weight_floor)

        pred_left = left_model.predict(X)
        pred_right = right_model.predict(X)
        residual_left = y - pred_left
        residual_right = y - pred_right

        n = y.shape[0]
        weighted_sq_error = (1.0 - pi) * residual_left**2 + pi * residual_right**2
        loss = float(np.mean(weighted_sq_error))

        coeff = (residual_right**2 - residual_left**2) * pi * (1.0 - pi) / n
        grad_w = X.T @ coeff / self.temperature
        grad_z = -float(np.sum(coeff)) / self.temperature
        grad = np.concatenate([grad_w, np.array([grad_z])])

        return SoftObjectiveResult(
            loss=loss,
            grad=grad,
            left_model=left_model,
            right_model=right_model,
            pi=pi,
            metadata={
                "temperature": self.temperature,
                "model_type": type(self.model_template).__name__,
                "mean_pi": float(np.mean(pi)),
                "min_pi": float(np.min(pi)),
                "max_pi": float(np.max(pi)),
            },
        )

    def value(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return self.value_and_grad(theta, X, y).loss
