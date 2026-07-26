"""Split optimizers for adaptive regression trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .metrics import mean_squared_error
from .models import (
    AffineRidgeModel,
    RegressionModel,
    WeightedRegressionModel,
    augment_features,
    scaled_ridge_from_gram,
)
from .objectives import SoftObliqueRidgeObjective
from .optimizers import AdaptiveAlpha, armijo_backtracking


HingeMode = Literal["max", "min", "both"]
ResolvedHingeMode = Literal["max", "min"]


class SplitNotFoundError(RuntimeError):
    """Raised when all splitter attempts produce an invalid hard partition."""

    def __init__(
        self,
        reason: str,
        message: str,
        diagnostics: dict[str, Any] | None = None,
        restarts_on_failure: int = 0,
        failure_reasons: tuple[str, ...] = (),
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics
        self.restarts_on_failure = int(restarts_on_failure)
        self.failure_reasons = tuple(failure_reasons)
        self.context = {} if context is None else dict(context)


@runtime_checkable
class Splitter(Protocol):
    """Common interface consumed by the tree builder."""

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel | None = None,
        parent_loss: float | None = None,
    ) -> "SplitResult":
        ...


def project_unit_w(theta: np.ndarray) -> np.ndarray:
    """Project theta = [w, z] onto ||w|| = 1, leaving z unchanged."""

    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size < 2:
        raise ValueError("theta must contain at least one w component and one z component.")
    w = theta[:-1]
    w_norm = np.linalg.norm(w)
    if w_norm <= 1e-12:
        raise ValueError("Cannot project theta with near-zero w.")
    projected = theta.copy()
    projected[:-1] = w / w_norm
    return projected


def project_unit_w_gradient(theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Project the w-gradient onto the tangent space of the unit sphere."""

    theta = project_unit_w(theta)
    grad = np.asarray(grad, dtype=float).reshape(-1)
    if grad.shape != theta.shape:
        raise ValueError(f"grad and theta must have the same shape, got {grad.shape} and {theta.shape}.")

    w = theta[:-1]
    grad_projected = grad.copy()
    grad_w = grad[:-1]
    grad_projected[:-1] = grad_w - float(grad_w @ w) * w
    return grad_projected


@dataclass
class SplitResult:
    """Result returned by a splitter."""

    w: np.ndarray
    z: float
    left_model: RegressionModel
    right_model: RegressionModel
    loss: float
    parent_loss: float
    split_gain: float
    relative_split_gain: float
    n_left: int
    n_right: int
    converged: bool
    n_iters: int
    stop_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the two hard-partition models."""

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError(f"X must have shape (n, {self.w.shape[0]}), got {X.shape}.")

        right = (X @ self.w - self.z) >= 0.0
        predictions = np.empty(X.shape[0], dtype=float)
        if np.any(~right):
            predictions[~right] = self.left_model.predict(X[~right])
        if np.any(right):
            predictions[right] = self.right_model.predict(X[right])
        return predictions


@dataclass
class SoftObliqueSplitter:
    """Soft sigmoid oblique splitter optimized by projected gradient descent."""

    model_template: WeightedRegressionModel
    temperature: float
    max_iters: int = 100
    grad_atol: float = 1e-8
    grad_rtol: float = 1e-5
    min_side_points: int = 2
    min_side_fraction: float = 0.0
    n_restarts: int = 1
    alpha0: float = 1.0
    rho: float = 0.5
    armijo_c: float = 1e-4
    max_backtracks: int = 25
    adaptive_alpha: bool = True
    alpha_min: float = 1e-12
    alpha_max: float = 1e3
    alpha_grow: float = 10.0
    alpha_recovery: float = 10.0
    heavy_backtrack_threshold: int = 8
    max_line_search_failures: int = 3
    weight_floor: float = 1e-12
    refit_during_line_search: bool = True
    random_state: int | np.random.Generator | None = None

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel | None = None,
        parent_loss: float | None = None,
    ) -> SplitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self._validate_inputs(X, y)

        rng = self._rng()
        required_side_points = self._required_side_points(X.shape[0])
        parent_model, parent_loss = self._parent_fit(X, y, parent_model, parent_loss)

        best: SplitResult | None = None
        last_failure: SplitNotFoundError | None = None
        for _ in range(max(1, self.n_restarts)):
            theta0 = self._initialize_theta(X, rng)
            try:
                result = self._optimize_from_initialization(
                    X=X,
                    y=y,
                    theta0=theta0,
                    parent_loss=parent_loss,
                    required_side_points=required_side_points,
                )
            except SplitNotFoundError as exc:
                last_failure = exc
                continue
            if best is None or result.loss < best.loss:
                best = result

        if best is None:
            if last_failure is not None:
                raise last_failure
            raise SplitNotFoundError(
                "invalid_split",
                "SoftObliqueSplitter could not find a nondegenerate split.",
            )
        return best

    def _optimize_from_initialization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta0: np.ndarray,
        parent_loss: float,
        required_side_points: int,
    ) -> SplitResult | None:
        objective = SoftObliqueRidgeObjective(
            temperature=self.temperature,
            model_template=self.model_template,
            weight_floor=self.weight_floor,
        )
        theta = project_unit_w(theta0)
        current = objective.value_and_grad(theta, X, y)
        if current.grad is None or current.loss is None:
            raise RuntimeError("Initial objective evaluation is incomplete.")
        projected_grad = project_unit_w_gradient(theta, current.grad)
        initial_grad_norm = float(np.linalg.norm(projected_grad))
        grad_tolerance = self.grad_atol + self.grad_rtol * initial_grad_norm
        loss_history = [current.loss]
        line_search_loss_history = []
        grad_norm_history = [initial_grad_norm]
        step_history = []
        backtrack_history = []
        alpha_controller = AdaptiveAlpha(
            alpha=self.alpha0,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            grow=self.alpha_grow,
            recovery=self.alpha_recovery,
            heavy_backtrack_threshold=self.heavy_backtrack_threshold,
        )
        alpha_fixed = float(self.alpha0)
        converged = False
        stop_reason = "max_iters"

        for _ in range(self.max_iters):
            if current.grad is None or current.loss is None:
                raise RuntimeError("Current objective evaluation is incomplete.")
            direction = -projected_grad
            directional_derivative = float(current.grad @ direction)
            grad_norm = grad_norm_history[-1]

            if grad_norm <= grad_tolerance:
                converged = True
                stop_reason = "gradient_tolerance"
                break
            if directional_derivative >= 0.0:
                stop_reason = "non_descent_direction"
                break

            line_search = None
            trial_evaluation = None
            failure_count = 0

            def candidate_value(candidate: np.ndarray) -> float:
                nonlocal trial_evaluation
                reference = None if self.refit_during_line_search else current
                trial_evaluation = objective.evaluate(
                    candidate,
                    X,
                    y,
                    reference=reference,
                )
                return objective.value(trial_evaluation)

            while True:
                alpha_start = alpha_controller.alpha if self.adaptive_alpha else alpha_fixed
                line_search = armijo_backtracking(
                    value_fn=candidate_value,
                    candidate_fn=lambda alpha: project_unit_w(theta + alpha * direction),
                    current_value=current.loss,
                    directional_derivative=directional_derivative,
                    alpha0=alpha_start,
                    rho=self.rho,
                    c=self.armijo_c,
                    max_backtracks=self.max_backtracks,
                )

                if self.adaptive_alpha:
                    alpha_controller.update(line_search, rho=self.rho)

                if line_search.success:
                    break

                failure_count += 1
                if not self.adaptive_alpha:
                    stop_reason = "line_search_failed"
                    break
                if failure_count >= self.max_line_search_failures:
                    stop_reason = "max_line_search_failures"
                    break
                if alpha_controller.alpha <= self.alpha_min:
                    stop_reason = "alpha_min_reached"
                    break

            if line_search is None or not line_search.success:
                break
            if trial_evaluation is None or not np.array_equal(
                trial_evaluation.theta,
                line_search.theta,
            ):
                raise RuntimeError(
                    "Accepted line-search candidate does not match its evaluation."
                )

            theta = line_search.theta
            line_search_loss_history.append(line_search.value)
            if self.refit_during_line_search:
                current = trial_evaluation
                objective.grad(current)
            else:
                current = objective.value_and_grad(theta, X, y)
            if current.grad is None or current.loss is None:
                raise RuntimeError("Accepted objective evaluation is incomplete.")
            loss_history.append(current.loss)
            projected_grad = project_unit_w_gradient(theta, current.grad)
            grad_norm_history.append(float(np.linalg.norm(projected_grad)))
            step_history.append(line_search.step_size)
            backtrack_history.append(line_search.n_backtracks)

        return self._make_hard_result(
            X=X,
            y=y,
            theta=theta,
            parent_loss=parent_loss,
            converged=converged,
            required_side_points=required_side_points,
            metadata={
                "soft_loss_history": loss_history,
                "line_search_loss_history": line_search_loss_history,
                "projected_grad_norm_history": grad_norm_history,
                "step_size_history": step_history,
                "backtrack_history": backtrack_history,
                "stop_reason": stop_reason,
                "temperature": self.temperature,
                "adaptive_alpha": self.adaptive_alpha,
                "refit_during_line_search": self.refit_during_line_search,
                "initial_grad_norm": initial_grad_norm,
                "final_grad_norm": grad_norm_history[-1],
                "grad_tolerance": grad_tolerance,
                "final_alpha": alpha_controller.alpha if self.adaptive_alpha else alpha_fixed,
                "alpha_min_saturated": bool(
                    self.adaptive_alpha and alpha_controller.alpha <= self.alpha_min
                ),
                "alpha_max_saturated": bool(
                    self.adaptive_alpha and alpha_controller.alpha >= self.alpha_max
                ),
                "final_soft_metadata": current.metadata,
            },
        )

    def _make_hard_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        parent_loss: float,
        converged: bool,
        required_side_points: int,
        metadata: dict[str, Any],
    ) -> SplitResult:
        theta = project_unit_w(theta)
        w = theta[:-1].copy()
        z = float(theta[-1])
        right_mask = (X @ w - z) >= 0.0
        n_right = int(np.sum(right_mask))
        n_left = int(right_mask.size - n_right)
        if n_left < required_side_points or n_right < required_side_points:
            raise SplitNotFoundError(
                "min_side_points",
                "SoftObliqueSplitter produced a partition below the minimum side count.",
                diagnostics={
                    **metadata,
                    "final_theta": theta.copy(),
                    "required_side_points": required_side_points,
                    "n_left": n_left,
                    "n_right": n_right,
                },
            )

        left_model = self.model_template.clone()
        right_model = self.model_template.clone()
        left_model.fit(X[~right_mask], y[~right_mask])
        right_model.fit(X[right_mask], y[right_mask])

        n = y.shape[0]
        left_loss = mean_squared_error(y[~right_mask], left_model.predict(X[~right_mask]))
        right_loss = mean_squared_error(y[right_mask], right_model.predict(X[right_mask]))
        loss = n_left / n * left_loss + n_right / n * right_loss

        metadata = {
            **metadata,
            "required_side_points": required_side_points,
            "left_loss": left_loss,
            "right_loss": right_loss,
        }
        return SplitResult(
            w=w,
            z=z,
            left_model=left_model,
            right_model=right_model,
            loss=loss,
            parent_loss=parent_loss,
            split_gain=parent_loss - loss,
            relative_split_gain=(parent_loss - loss) / max(parent_loss, 1e-12),
            n_left=n_left,
            n_right=n_right,
            converged=converged,
            n_iters=len(metadata["soft_loss_history"]) - 1,
            stop_reason=str(metadata["stop_reason"]),
            metadata=metadata,
        )

    def _initialize_theta(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        w = rng.normal(size=X.shape[1])
        w_norm = np.linalg.norm(w)
        if w_norm <= 1e-12:
            w[0] = 1.0
            w_norm = 1.0
        w = w / w_norm
        z = float(np.median(X @ w))
        return np.concatenate([w, np.array([z])])

    def _parent_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel | None,
        parent_loss: float | None,
    ) -> tuple[RegressionModel, float]:
        if parent_model is None:
            parent_model = self.model_template.clone()
            parent_model.fit(X, y)
        if parent_loss is None:
            parent_loss = mean_squared_error(y, parent_model.predict(X))
        return parent_model, float(parent_loss)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.max_iters < 0:
            raise ValueError("max_iters must be nonnegative.")
        if self.grad_atol < 0.0:
            raise ValueError("grad_atol must be nonnegative.")
        if self.grad_rtol < 0.0:
            raise ValueError("grad_rtol must be nonnegative.")
        if self.min_side_points < 1:
            raise ValueError("min_side_points must be at least 1.")
        if not (0.0 <= self.min_side_fraction < 0.5):
            raise ValueError("min_side_fraction must satisfy 0 <= fraction < 0.5.")
        if self.max_line_search_failures < 1:
            raise ValueError("max_line_search_failures must be at least 1.")

    def _required_side_points(self, n_total: int) -> int:
        fraction_count = int(np.ceil(self.min_side_fraction * n_total))
        return max(int(self.min_side_points), fraction_count)

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)


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
        last_failure: SplitNotFoundError | None = None
        required_side_points = self._required_side_points(X.shape[0])
        modes: tuple[ResolvedHingeMode, ...] = ("max", "min") if self.mode == "both" else (self.mode,)
        for mode in modes:
            for _ in range(max(1, self.n_restarts)):
                theta1, theta2 = self._initialize_thetas(parent_theta, rng)
                try:
                    result = self._fit_from_initialization(
                        X_aug,
                        y,
                        theta1,
                        theta2,
                        parent_loss,
                        mode,
                        required_side_points,
                    )
                except SplitNotFoundError as exc:
                    last_failure = exc
                    continue
                if best is None or result.loss < best.loss:
                    best = result

        if best is None:
            if last_failure is not None:
                raise last_failure
            raise SplitNotFoundError(
                "invalid_split",
                "HingeAffineSplitter could not find a nondegenerate split.",
            )
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
    ) -> SplitResult:
        loss_history = []
        previous_active1 = None
        converged = False

        for iteration in range(self.max_iters):
            active1 = self._active_theta1(X_aug, theta1, theta2, mode)
            n1 = int(np.sum(active1))
            n2 = int(active1.size - n1)
            if n1 < required_side_points or n2 < required_side_points:
                raise SplitNotFoundError(
                    "min_side_points",
                    "HingeAffineSplitter produced an intermediate partition below the minimum side count.",
                    diagnostics={
                        "mode": mode,
                        "theta1": theta1.copy(),
                        "theta2": theta2.copy(),
                        "loss_history": loss_history,
                        "required_side_points": required_side_points,
                        "n_left": n1,
                        "n_right": n2,
                        "iteration": iteration,
                        "failure_stage": "intermediate_partition",
                    },
                )

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
    ) -> SplitResult:
        delta = theta1 - theta2
        w = delta[:-1].copy()
        z = -float(delta[-1])
        w_norm = float(np.linalg.norm(w))
        if w_norm <= 1e-12:
            raise SplitNotFoundError(
                "invalid_split",
                "HingeAffineSplitter produced a near-zero boundary normal.",
                diagnostics={
                    "mode": mode,
                    "theta1": theta1.copy(),
                    "theta2": theta2.copy(),
                    "loss_history": loss_history,
                    "required_side_points": required_side_points,
                    "stop_reason": "converged" if converged else "max_iters",
                    "failure_stage": "boundary_normal",
                },
            )
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
            raise SplitNotFoundError(
                "min_side_points",
                "HingeAffineSplitter produced a final partition below the minimum side count.",
                diagnostics={
                    "mode": mode,
                    "theta1": theta1.copy(),
                    "theta2": theta2.copy(),
                    "loss_history": loss_history,
                    "required_side_points": required_side_points,
                    "n_left": n_left,
                    "n_right": n_right,
                    "stop_reason": "converged" if converged else "max_iters",
                    "failure_stage": "final_partition",
                },
            )

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
            relative_split_gain=(parent_loss - loss) / max(parent_loss, 1e-12),
            n_left=n_left,
            n_right=n_right,
            converged=converged,
            n_iters=len(loss_history),
            stop_reason="converged" if converged else "max_iters",
            metadata={
                "mode": mode,
                "theta1": theta1.copy(),
                "theta2": theta2.copy(),
                "loss_history": loss_history,
                "required_side_points": required_side_points,
                "stop_reason": "converged" if converged else "max_iters",
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
