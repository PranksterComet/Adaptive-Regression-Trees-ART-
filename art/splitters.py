"""Split optimizers for adaptive regression trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .metrics import mean_squared_error
from .models import AffineRidgeModel, augment_features, scaled_ridge_from_gram


HingeMode = Literal["max", "min", "both"]
ResolvedHingeMode = Literal["max", "min"]


@dataclass
class SplitResult:
    """Result returned by a splitter."""

    w: np.ndarray
    z: float
    left_model: AffineRidgeModel
    right_model: AffineRidgeModel
    loss: float
    parent_loss: float
    split_gain: float
    n_left: int
    n_right: int
    converged: bool
    n_iters: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HingeAffineSplitter:
    """Hinge regression split using two affine ridge fits.

    The fitted node model is max(l1, l2) or min(l1, l2), where each l_j is
    affine. The induced tree boundary is l1(x) = l2(x).
    """

    mode: HingeMode = "both"
    ridge: float = 1e-8
    mu: float = 1.0
    max_iters: int = 100
    tol: float = 1e-6
    min_side_points: int = 2
    min_side_fraction: float = 0.0
    n_restarts: int = 5
    init_scale: float = 1e-2
    random_state: int | np.random.Generator | None = None

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: AffineRidgeModel | None = None,
        parent_loss: float | None = None,
    ) -> SplitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}.")
        if self.mode not in ("max", "min", "both"):
            raise ValueError("mode must be 'max', 'min', or 'both'.")
        if not (0.0 < self.mu <= 1.0):
            raise ValueError("mu must satisfy 0 < mu <= 1.")
        if self.min_side_points < 1:
            raise ValueError("min_side_points must be at least 1.")
        if not (0.0 <= self.min_side_fraction < 0.5):
            raise ValueError("min_side_fraction must satisfy 0 <= fraction < 0.5.")

        rng = self._rng()
        X_aug = augment_features(X)
        parent_theta = self._parent_theta(X_aug, y, parent_model)
        if parent_loss is None:
            parent_loss = mean_squared_error(y, X_aug @ parent_theta)
        else:
            parent_loss = float(parent_loss)

        best: SplitResult | None = None
        required_side_points = self._required_side_points(X.shape[0])
        modes: tuple[ResolvedHingeMode, ...] = ("max", "min") if self.mode == "both" else (self.mode,)
        for mode in modes:
            for _ in range(max(1, self.n_restarts)):
                theta1, theta2 = self._initialize_thetas(parent_theta, rng)
                result = self._fit_from_initialization(
                    X_aug,
                    y,
                    theta1,
                    theta2,
                    parent_loss,
                    mode,
                    required_side_points,
                )
                if result is not None and (best is None or result.loss < best.loss):
                    best = result

        if best is None:
            raise RuntimeError("HingeAffineSplitter could not find a nondegenerate split.")
        return best

    def _fit_from_initialization(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        parent_loss: float,
        mode: ResolvedHingeMode,
        required_side_points: int,
    ) -> SplitResult | None:
        loss_history = []
        previous_active1 = None
        converged = False

        for iteration in range(self.max_iters):
            active1 = self._active_theta1(X_aug, theta1, theta2, mode)
            n1 = int(np.sum(active1))
            n2 = int(active1.size - n1)
            if n1 < required_side_points or n2 < required_side_points:
                return None

            theta1_target = self._fit_theta(X_aug[active1], y[active1])
            theta2_target = self._fit_theta(X_aug[~active1], y[~active1])
            old_theta = np.concatenate([theta1, theta2])

            theta1 = theta1 + self.mu * (theta1_target - theta1)
            theta2 = theta2 + self.mu * (theta2_target - theta2)

            pred = self._hinge_predict(X_aug, theta1, theta2, mode)
            loss = mean_squared_error(y, pred)
            loss_history.append(loss)

            new_theta = np.concatenate([theta1, theta2])
            theta_step = np.linalg.norm(new_theta - old_theta) / max(np.linalg.norm(old_theta), 1.0)
            partition_stable = previous_active1 is not None and np.array_equal(active1, previous_active1)
            if theta_step <= self.tol and partition_stable:
                converged = True
                break
            previous_active1 = active1

        return self._make_result(
            X_aug=X_aug,
            y=y,
            theta1=theta1,
            theta2=theta2,
            parent_loss=parent_loss,
            loss_history=loss_history,
            converged=converged,
            mode=mode,
            required_side_points=required_side_points,
        )

    def _make_result(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        parent_loss: float,
        loss_history: list[float],
        converged: bool,
        mode: ResolvedHingeMode,
        required_side_points: int,
    ) -> SplitResult | None:
        delta = theta1 - theta2
        w = delta[:-1].copy()
        z = -float(delta[-1])
        w_norm = float(np.linalg.norm(w))
        if w_norm <= 1e-12:
            return None
        w = w / w_norm
        z = z / w_norm

        scores = X_aug @ delta
        right_mask = scores >= 0.0
        if mode == "max":
            left_theta = theta2
            right_theta = theta1
        else:
            left_theta = theta1
            right_theta = theta2

        n_right = int(np.sum(right_mask))
        n_left = int(right_mask.size - n_right)
        if n_left < required_side_points or n_right < required_side_points:
            return None

        left_model = self._model_from_theta(left_theta)
        right_model = self._model_from_theta(right_theta)
        left_pred = X_aug[~right_mask] @ left_model.coef_
        right_pred = X_aug[right_mask] @ right_model.coef_
        n = y.shape[0]
        loss = (
            n_left / n * mean_squared_error(y[~right_mask], left_pred)
            + n_right / n * mean_squared_error(y[right_mask], right_pred)
        )

        return SplitResult(
            w=w,
            z=z,
            left_model=left_model,
            right_model=right_model,
            loss=loss,
            parent_loss=parent_loss,
            split_gain=parent_loss - loss,
            n_left=n_left,
            n_right=n_right,
            converged=converged,
            n_iters=len(loss_history),
            metadata={
                "mode": mode,
                "theta1": theta1.copy(),
                "theta2": theta2.copy(),
                "loss_history": loss_history,
                "required_side_points": required_side_points,
            },
        )

    def _initialize_thetas(
        self,
        parent_theta: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = rng.normal(size=parent_theta.shape[0])
        delta_norm = np.linalg.norm(delta)
        if delta_norm <= 1e-12:
            delta[-1] = 1.0
            delta_norm = 1.0
        scale = self.init_scale * max(float(np.linalg.norm(parent_theta)), 1.0)
        delta = scale * delta / delta_norm
        return parent_theta + delta, parent_theta - delta

    def _fit_theta(self, X_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
        gram = X_aug.T @ X_aug
        rhs = X_aug.T @ y
        ridge_eff = scaled_ridge_from_gram(gram, self.ridge)
        return np.linalg.solve(gram + ridge_eff * np.eye(gram.shape[0]), rhs)

    def _parent_theta(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        parent_model: AffineRidgeModel | None,
    ) -> np.ndarray:
        if parent_model is None:
            return self._fit_theta(X_aug, y)
        if parent_model.coef_ is None:
            raise ValueError("parent_model must be fit before being passed to the splitter.")
        theta = np.asarray(parent_model.coef_, dtype=float).reshape(-1)
        if theta.shape[0] != X_aug.shape[1]:
            raise ValueError(f"parent_model coefficient length must be {X_aug.shape[1]}, got {theta.shape[0]}.")
        return theta.copy()

    def _active_theta1(
        self,
        X_aug: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        mode: ResolvedHingeMode,
    ) -> np.ndarray:
        scores = X_aug @ (theta1 - theta2)
        if mode == "max":
            return scores >= 0.0
        return scores <= 0.0

    def _hinge_predict(
        self,
        X_aug: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        mode: ResolvedHingeMode,
    ) -> np.ndarray:
        pred1 = X_aug @ theta1
        pred2 = X_aug @ theta2
        if mode == "max":
            return np.maximum(pred1, pred2)
        return np.minimum(pred1, pred2)

    def _model_from_theta(self, theta: np.ndarray) -> AffineRidgeModel:
        model = AffineRidgeModel(ridge=self.ridge)
        model.coef_ = theta.copy()
        return model

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    def _required_side_points(self, n_total: int) -> int:
        fraction_count = int(np.ceil(self.min_side_fraction * n_total))
        return max(int(self.min_side_points), fraction_count)
