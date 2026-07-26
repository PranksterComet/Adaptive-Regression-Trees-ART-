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
    """Cached evaluation state for one soft-oblique candidate."""

    theta: np.ndarray
    X: np.ndarray
    left_model: RegressionModel
    right_model: RegressionModel
    pi: np.ndarray
    residual_left: np.ndarray
    residual_right: np.ndarray
    loss: float | None = None
    grad: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftObliqueRidgeObjective:
    """Soft sigmoid-gated objective for weighted-fit-compatible models."""

    temperature: float
    model_template: WeightedRegressionModel
    weight_floor: float = 1e-12

    def evaluate(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        reference: SoftObjectiveResult | None = None,
    ) -> SoftObjectiveResult:
        """Evaluate shared candidate state, optionally reusing fitted models."""

        theta, X, y = self._validate_inputs(theta, X, y)
        w = theta[:-1]
        z = float(theta[-1])
        pi = sigmoid((X @ w - z) / self.temperature)

        if reference is None:
            left_model, right_model = self._fit_models(X, y, pi)
        else:
            left_model = reference.left_model
            right_model = reference.right_model

        residual_left = y - left_model.predict(X)
        residual_right = y - right_model.predict(X)
        return SoftObjectiveResult(
            theta=theta.copy(),
            X=X,
            left_model=left_model,
            right_model=right_model,
            pi=pi,
            residual_left=residual_left,
            residual_right=residual_right,
            metadata={
                "temperature": self.temperature,
                "model_type": type(self.model_template).__name__,
                "mean_pi": float(np.mean(pi)),
                "min_pi": float(np.min(pi)),
                "max_pi": float(np.max(pi)),
                "models_refit": reference is None,
            },
        )

    def value(
        self,
        evaluation_or_theta: SoftObjectiveResult | np.ndarray,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> float:
        """Return a cached loss or evaluate one from theta, X, and y."""

        evaluation = self._resolve_evaluation(evaluation_or_theta, X, y)
        if evaluation.loss is None:
            evaluation.loss = self._loss(
                evaluation.pi,
                evaluation.residual_left,
                evaluation.residual_right,
            )
        return evaluation.loss

    def grad(self, evaluation: SoftObjectiveResult) -> np.ndarray:
        """Return the cached gradient or calculate it from stored candidate state."""

        if not isinstance(evaluation, SoftObjectiveResult):
            raise TypeError("evaluation must be a SoftObjectiveResult.")
        if evaluation.grad is None:
            evaluation.grad = self._gradient(
                evaluation.X,
                evaluation.pi,
                evaluation.residual_left,
                evaluation.residual_right,
            )
        return evaluation.grad

    def value_and_grad(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        reference: SoftObjectiveResult | None = None,
    ) -> SoftObjectiveResult:
        """Evaluate a candidate and populate both cached loss and gradient."""

        evaluation = self.evaluate(theta, X, y, reference=reference)
        self.value(evaluation)
        self.grad(evaluation)
        return evaluation

    def _fit_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pi: np.ndarray,
    ) -> tuple[RegressionModel, RegressionModel]:
        left_model = self.model_template.clone()
        right_model = self.model_template.clone()
        left_model.fit_weighted(
            X,
            y,
            weights=1.0 - pi,
            weight_floor=self.weight_floor,
        )
        right_model.fit_weighted(
            X,
            y,
            weights=pi,
            weight_floor=self.weight_floor,
        )
        return left_model, right_model

    def _loss(
        self,
        pi: np.ndarray,
        residual_left: np.ndarray,
        residual_right: np.ndarray,
    ) -> float:
        weighted_sq_error = (
            (1.0 - pi) * residual_left**2 + pi * residual_right**2
        )
        return float(np.mean(weighted_sq_error))

    def _gradient(
        self,
        X: np.ndarray,
        pi: np.ndarray,
        residual_left: np.ndarray,
        residual_right: np.ndarray,
    ) -> np.ndarray:
        n = X.shape[0]
        coeff = (
            (residual_right**2 - residual_left**2)
            * pi
            * (1.0 - pi)
            / n
        )
        grad_w = X.T @ coeff / self.temperature
        grad_z = -float(np.sum(coeff)) / self.temperature
        return np.concatenate([grad_w, np.array([grad_z])])

    def _resolve_evaluation(
        self,
        evaluation_or_theta: SoftObjectiveResult | np.ndarray,
        X: np.ndarray | None,
        y: np.ndarray | None,
    ) -> SoftObjectiveResult:
        if isinstance(evaluation_or_theta, SoftObjectiveResult):
            if X is not None or y is not None:
                raise ValueError("X and y must be omitted when passing an evaluation.")
            return evaluation_or_theta
        if X is None or y is None:
            raise ValueError("X and y are required when passing theta.")
        return self.evaluate(evaluation_or_theta, X, y)

    def _validate_inputs(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return theta, X, y
