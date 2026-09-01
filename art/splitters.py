"""Split optimizers for adaptive regression trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .metrics import mean_squared_error
from .models import (
    AffineRidgeModel,
    MODEL_CONDITION_WARNING_THRESHOLD,
    PreparedDesign,
    PreparedFeatureModel,
    RegressionModel,
    RidgeSolveResult,
    RidgeSolver,
    WeightedRegressionModel,
    augment_features,
    ridge_solve_diagnostics,
    solve_weighted_ridge,
)
from .objectives import SoftObliqueRidgeObjective
from .optimizers import AdaptiveAlpha, armijo_backtracking
from .timing import BuildTimingProfile


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
        prepared_design: PreparedDesign | None = None,
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


def _solve_field(
    evaluation: Any,
    side: Literal["left", "right"],
    field_name: str,
) -> object | None:
    diagnostics = evaluation.metadata.get(f"{side}_solve")
    return diagnostics.get(field_name) if isinstance(diagnostics, dict) else None


def _is_high_condition(value: object) -> bool:
    return value is not None and float(value) >= MODEL_CONDITION_WARNING_THRESHOLD


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

    def predict_prepared(
        self,
        X: np.ndarray,
        prepared_design: PreparedDesign,
    ) -> np.ndarray:
        """Predict from raw routing coordinates and an aligned design matrix."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError(f"X must have shape (n, {self.w.shape[0]}), got {X.shape}.")
        if prepared_design.n_samples != X.shape[0]:
            raise ValueError("Prepared design and X must have the same number of rows.")
        if prepared_design.input_dimension != X.shape[1]:
            raise ValueError("Prepared design input dimension does not match X.")
        if not isinstance(self.left_model, PreparedFeatureModel) or not isinstance(
            self.right_model, PreparedFeatureModel
        ):
            raise TypeError(
                "predict_prepared requires prepared-feature split models."
            )

        right = (X @ self.w - self.z) >= 0.0
        predictions = np.empty(X.shape[0], dtype=float)
        if np.any(~right):
            predictions[~right] = self.left_model.predict_design(
                prepared_design.subset(~right)
            )
        if np.any(right):
            predictions[right] = self.right_model.predict_design(
                prepared_design.subset(right)
            )
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
    refit_during_line_search: bool = False
    random_state: int | np.random.Generator | None = None
    _timing_profile: BuildTimingProfile | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel | None = None,
        parent_loss: float | None = None,
        prepared_design: PreparedDesign | None = None,
    ) -> SplitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self._validate_inputs(X, y)
        if prepared_design is None and isinstance(
            self.model_template, PreparedFeatureModel
        ):
            prepared_design = self.model_template.prepare_design(X)
        if prepared_design is not None:
            if prepared_design.n_samples != X.shape[0]:
                raise ValueError(
                    "Prepared design and X must have the same number of rows."
                )
            if prepared_design.input_dimension != X.shape[1]:
                raise ValueError("Prepared design input dimension does not match X.")

        rng = self._rng()
        required_side_points = self._required_side_points(X.shape[0])
        parent_model, parent_loss = self._parent_fit(
            X,
            y,
            parent_model,
            parent_loss,
            prepared_design,
        )
        parent_solve = ridge_solve_diagnostics(parent_model)
        objective = SoftObliqueRidgeObjective(
            temperature=self.temperature,
            model_template=self.model_template,
            X=X,
            y=y,
            weight_floor=self.weight_floor,
            prepared_design=prepared_design,
            timing_profile=self._timing_profile,
        )

        best: SplitResult | None = None
        last_failure: SplitNotFoundError | None = None
        for _ in range(max(1, self.n_restarts)):
            theta0 = self._initialize_theta(X, rng)
            try:
                result = self._optimize_from_initialization(
                    objective=objective,
                    theta0=theta0,
                    parent_loss=parent_loss,
                    required_side_points=required_side_points,
                    parent_solve=parent_solve,
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
        objective: SoftObliqueRidgeObjective,
        theta0: np.ndarray,
        parent_loss: float,
        required_side_points: int,
        parent_solve: dict[str, object] | None,
    ) -> SplitResult | None:
        theta = project_unit_w(theta0)
        current = objective.value_and_grad(theta)
        if current.grad is None or current.loss is None:
            raise RuntimeError("Initial objective evaluation is incomplete.")
        projected_grad = project_unit_w_gradient(theta, current.grad)
        initial_grad_norm = float(np.linalg.norm(projected_grad))
        grad_tolerance = self.grad_atol + self.grad_rtol * initial_grad_norm
        loss_history = [current.loss]
        line_search_loss_history = []
        grad_norm_history = [initial_grad_norm]
        left_cond_history = [_solve_field(current, "left", "cond_estimate")]
        right_cond_history = [_solve_field(current, "right", "cond_estimate")]
        left_condition_estimator_history = [
            _solve_field(current, "left", "condition_estimator")
        ]
        right_condition_estimator_history = [
            _solve_field(current, "right", "condition_estimator")
        ]
        track_solver_history = getattr(self.model_template, "solver", None) == "auto"
        left_solver_history = (
            [_solve_field(current, "left", "solver_used")]
            if track_solver_history
            else None
        )
        right_solver_history = (
            [_solve_field(current, "right", "solver_used")]
            if track_solver_history
            else None
        )
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
                current = objective.value_and_grad(theta)
            if current.grad is None or current.loss is None:
                raise RuntimeError("Accepted objective evaluation is incomplete.")
            loss_history.append(current.loss)
            projected_grad = project_unit_w_gradient(theta, current.grad)
            grad_norm_history.append(float(np.linalg.norm(projected_grad)))
            left_cond_history.append(
                _solve_field(current, "left", "cond_estimate")
            )
            right_cond_history.append(
                _solve_field(current, "right", "cond_estimate")
            )
            left_condition_estimator_history.append(
                _solve_field(current, "left", "condition_estimator")
            )
            right_condition_estimator_history.append(
                _solve_field(current, "right", "condition_estimator")
            )
            if left_solver_history is not None and right_solver_history is not None:
                left_solver_history.append(
                    _solve_field(current, "left", "solver_used")
                )
                right_solver_history.append(
                    _solve_field(current, "right", "solver_used")
                )
            step_history.append(line_search.step_size)
            backtrack_history.append(line_search.n_backtracks)

        conditioning_values = [
            parent_solve.get("cond_estimate") if parent_solve is not None else None,
            *left_cond_history,
            *right_cond_history,
        ]
        solver_history_metadata = {}
        if left_solver_history is not None and right_solver_history is not None:
            solver_history_metadata = {
                "left_solver_history": left_solver_history,
                "right_solver_history": right_solver_history,
            }
        return self._make_hard_result(
            objective=objective,
            theta=theta,
            parent_loss=parent_loss,
            converged=converged,
            required_side_points=required_side_points,
            metadata={
                "soft_loss_history": loss_history,
                "line_search_loss_history": line_search_loss_history,
                "projected_grad_norm_history": grad_norm_history,
                "left_cond_estimate_history": left_cond_history,
                "right_cond_estimate_history": right_cond_history,
                "left_condition_estimator_history": left_condition_estimator_history,
                "right_condition_estimator_history": right_condition_estimator_history,
                **solver_history_metadata,
                "step_size_history": step_history,
                "backtrack_history": backtrack_history,
                "stop_reason": stop_reason,
                "temperature": self.temperature,
                "adaptive_alpha": self.adaptive_alpha,
                "refit_during_line_search": self.refit_during_line_search,
                "initial_grad_norm": initial_grad_norm,
                "final_grad_norm": grad_norm_history[-1],
                "n_iters": len(loss_history) - 1,
                "grad_tolerance": grad_tolerance,
                "final_alpha": alpha_controller.alpha if self.adaptive_alpha else alpha_fixed,
                "alpha_min_saturated": bool(
                    self.adaptive_alpha and alpha_controller.alpha <= self.alpha_min
                ),
                "alpha_max_saturated": bool(
                    self.adaptive_alpha and alpha_controller.alpha >= self.alpha_max
                ),
                "final_soft_metadata": current.metadata,
                "parent_solve": parent_solve,
                "high_model_conditioning": any(
                    _is_high_condition(value) for value in conditioning_values
                ),
            },
        )

    def _make_hard_result(
        self,
        objective: SoftObliqueRidgeObjective,
        theta: np.ndarray,
        parent_loss: float,
        converged: bool,
        required_side_points: int,
        metadata: dict[str, Any],
    ) -> SplitResult:
        X = objective.X
        y = objective.y
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
        if objective.prepared_design is None:
            left_model.fit(X[~right_mask], y[~right_mask])
            right_model.fit(X[right_mask], y[right_mask])
            left_predictions = left_model.predict(X[~right_mask])
            right_predictions = right_model.predict(X[right_mask])
        else:
            if not isinstance(left_model, PreparedFeatureModel) or not isinstance(
                right_model, PreparedFeatureModel
            ):
                raise TypeError(
                    "Prepared hard-split models must implement PreparedFeatureModel."
                )
            left_design = objective.prepared_design.subset(~right_mask)
            right_design = objective.prepared_design.subset(right_mask)
            left_model.fit_design(left_design, y[~right_mask])
            right_model.fit_design(right_design, y[right_mask])
            left_predictions = left_model.predict_design(left_design)
            right_predictions = right_model.predict_design(right_design)

        n = y.shape[0]
        left_loss = mean_squared_error(y[~right_mask], left_predictions)
        right_loss = mean_squared_error(y[right_mask], right_predictions)
        loss = n_left / n * left_loss + n_right / n * right_loss

        metadata = {
            **metadata,
            "required_side_points": required_side_points,
            "left_loss": left_loss,
            "right_loss": right_loss,
            "hard_left_solve": ridge_solve_diagnostics(left_model),
            "hard_right_solve": ridge_solve_diagnostics(right_model),
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
            n_iters=int(metadata["n_iters"]),
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
        prepared_design: PreparedDesign | None,
    ) -> tuple[RegressionModel, float]:
        if parent_model is None:
            parent_model = self.model_template.clone()
            if prepared_design is None:
                parent_model.fit(X, y)
            else:
                if not isinstance(parent_model, PreparedFeatureModel):
                    raise TypeError(
                        "Prepared parent model must implement PreparedFeatureModel."
                    )
                parent_model.fit_design(prepared_design, y)
        if parent_loss is None:
            if prepared_design is None:
                predictions = parent_model.predict(X)
            else:
                if not isinstance(parent_model, PreparedFeatureModel):
                    raise TypeError(
                        "Prepared parent model must implement PreparedFeatureModel."
                    )
                predictions = parent_model.predict_design(prepared_design)
            parent_loss = mean_squared_error(y, predictions)
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

    HRT optimizes max(l1, l2) or min(l1, l2), where each l_j is affine, and
    uses l1(x) = l2(x) as the boundary. The returned child models are fresh
    hard-partition fits, independent of which hinge mode found the boundary.
    """

    mode: HingeMode = "both"
    ridge: float = 1e-8
    solver: RidgeSolver = "normal"
    auto_rcond_threshold: float = 1e-10
    mu: float = 1.0
    max_iters: int = 100
    tol: float = 1e-6
    min_side_points: int = 2
    min_side_fraction: float = 0.0
    n_restarts: int = 5
    init_scale: float = 1e-2
    random_state: int | np.random.Generator | None = None

    def __post_init__(self) -> None:
        # Reuse the affine model's validation and preload its solver dependency.
        AffineRidgeModel(
            ridge=self.ridge,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )

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
        parent_theta, parent_solve = self._parent_theta(X_aug, y, parent_model)
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
                theta1, theta2 = self._initialize_thetas(
                    X_aug,
                    parent_theta,
                    rng,
                )
                try:
                    result = self._fit_from_initialization(
                        X_aug,
                        y,
                        theta1,
                        theta2,
                        parent_loss,
                        mode,
                        required_side_points,
                        parent_solve,
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
        parent_solve: dict[str, object] | None,
    ) -> SplitResult:
        loss_history = []
        previous_active1 = None
        converged = False
        diagnostics: dict[str, Any] = {
            "mode": mode,
            "loss_history": loss_history,
            "left_cond_estimate_history": [],
            "right_cond_estimate_history": [],
            "left_condition_estimator_history": [],
            "right_condition_estimator_history": [],
            "parent_solve": parent_solve,
            "high_model_conditioning": bool(
                parent_solve is not None
                and _is_high_condition(parent_solve.get("cond_estimate"))
            ),
            "required_side_points": required_side_points,
        }
        if self.solver == "auto":
            diagnostics["left_solver_history"] = []
            diagnostics["right_solver_history"] = []

        for iteration in range(self.max_iters):
            active1 = self._active_theta1(X_aug, theta1, theta2, mode)
            n1 = int(np.sum(active1))
            n2 = int(active1.size - n1)
            n_left, n_right = (n2, n1) if mode == "max" else (n1, n2)
            if iteration == 0:
                diagnostics["initial_n_left"] = n_left
                diagnostics["initial_n_right"] = n_right
            if n1 < required_side_points or n2 < required_side_points:
                raise SplitNotFoundError(
                    "min_side_points",
                    "HingeAffineSplitter produced an intermediate partition below the minimum side count.",
                    diagnostics={
                        **diagnostics,
                        "theta1": theta1.copy(),
                        "theta2": theta2.copy(),
                        "n_left": n_left,
                        "n_right": n_right,
                        "n_iters": len(loss_history),
                        "stop_reason": "min_side_points",
                        "iteration": iteration,
                        "failure_stage": "intermediate_partition",
                    },
                )

            theta1_solve = self._fit_theta(X_aug[active1], y[active1])
            theta2_solve = self._fit_theta(X_aug[~active1], y[~active1])
            if mode == "max":
                self._append_solve_diagnostics(diagnostics, "right", theta1_solve)
                self._append_solve_diagnostics(diagnostics, "left", theta2_solve)
            else:
                self._append_solve_diagnostics(diagnostics, "left", theta1_solve)
                self._append_solve_diagnostics(diagnostics, "right", theta2_solve)
            theta1_target = theta1_solve.coefficients
            theta2_target = theta2_solve.coefficients
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

        diagnostics.update(
            {
                "theta1": theta1.copy(),
                "theta2": theta2.copy(),
                "n_iters": len(loss_history),
                "stop_reason": "converged" if converged else "max_iters",
            }
        )
        return self._make_result(
            X_aug=X_aug,
            y=y,
            theta1=theta1,
            theta2=theta2,
            parent_loss=parent_loss,
            converged=converged,
            mode=mode,
            required_side_points=required_side_points,
            diagnostics=diagnostics,
        )

    def _make_result(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        parent_loss: float,
        converged: bool,
        mode: ResolvedHingeMode,
        required_side_points: int,
        diagnostics: dict[str, Any],
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
                    **diagnostics,
                    "theta1": theta1.copy(),
                    "theta2": theta2.copy(),
                    "stop_reason": "converged" if converged else "max_iters",
                    "failure_stage": "boundary_normal",
                },
            )
        w = w / w_norm
        z = z / w_norm

        scores = X_aug @ delta
        right_mask = scores >= 0.0

        n_right = int(np.sum(right_mask))
        n_left = int(right_mask.size - n_right)
        if n_left < required_side_points or n_right < required_side_points:
            raise SplitNotFoundError(
                "min_side_points",
                "HingeAffineSplitter produced a final partition below the minimum side count.",
                diagnostics={
                    **diagnostics,
                    "theta1": theta1.copy(),
                    "theta2": theta2.copy(),
                    "n_left": n_left,
                    "n_right": n_right,
                    "stop_reason": "converged" if converged else "max_iters",
                    "failure_stage": "final_partition",
                },
            )

        left_solve = self._fit_theta(X_aug[~right_mask], y[~right_mask])
        right_solve = self._fit_theta(X_aug[right_mask], y[right_mask])
        left_model = self._model_from_solve(left_solve)
        right_model = self._model_from_solve(right_solve)
        left_pred = X_aug[~right_mask] @ left_solve.coefficients
        right_pred = X_aug[right_mask] @ right_solve.coefficients
        hard_left_solve = ridge_solve_diagnostics(left_solve)
        hard_right_solve = ridge_solve_diagnostics(right_solve)
        diagnostics["hard_left_solve"] = hard_left_solve
        diagnostics["hard_right_solve"] = hard_right_solve
        if (
            hard_left_solve is not None
            and _is_high_condition(hard_left_solve.get("cond_estimate"))
        ) or (
            hard_right_solve is not None
            and _is_high_condition(hard_right_solve.get("cond_estimate"))
        ):
            diagnostics["high_model_conditioning"] = True
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
            n_iters=int(diagnostics["n_iters"]),
            stop_reason="converged" if converged else "max_iters",
            metadata={
                **diagnostics,
                "n_left": n_left,
                "n_right": n_right,
            },
        )

    def _initialize_thetas(
        self,
        X_aug: np.ndarray,
        parent_theta: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        normal = rng.normal(size=X_aug.shape[1] - 1)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            normal[0] = 1.0
            normal_norm = 1.0
        normal /= normal_norm
        offset = float(np.median(X_aug[:, :-1] @ normal))
        boundary = np.concatenate([normal, [-offset]])
        scale = self.init_scale * max(float(np.linalg.norm(parent_theta)), 1.0)
        delta = scale * boundary / np.linalg.norm(boundary)
        return parent_theta + delta, parent_theta - delta

    def _fit_theta(self, X_aug: np.ndarray, y: np.ndarray) -> RidgeSolveResult:
        return solve_weighted_ridge(
            X_aug,
            y,
            np.ones(y.shape[0], dtype=float),
            ridge=self.ridge,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )

    def _append_solve_diagnostics(
        self,
        diagnostics: dict[str, Any],
        side: Literal["left", "right"],
        result: RidgeSolveResult,
    ) -> None:
        solve = ridge_solve_diagnostics(result)
        if solve is None:
            raise RuntimeError("HRT ridge solve did not produce diagnostics.")
        diagnostics[f"{side}_cond_estimate_history"].append(
            solve["cond_estimate"]
        )
        diagnostics[f"{side}_condition_estimator_history"].append(
            solve["condition_estimator"]
        )
        if self.solver == "auto":
            diagnostics[f"{side}_solver_history"].append(solve["solver_used"])
        if _is_high_condition(solve["cond_estimate"]):
            diagnostics["high_model_conditioning"] = True

    def _parent_theta(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        parent_model: AffineRidgeModel | None,
    ) -> tuple[np.ndarray, dict[str, object] | None]:
        if parent_model is None:
            result = self._fit_theta(X_aug, y)
            return result.coefficients, ridge_solve_diagnostics(result)
        if parent_model.coef_ is None:
            raise ValueError("parent_model must be fit before being passed to the splitter.")
        theta = np.asarray(parent_model.coef_, dtype=float).reshape(-1)
        if theta.shape[0] != X_aug.shape[1]:
            raise ValueError(f"parent_model coefficient length must be {X_aug.shape[1]}, got {theta.shape[0]}.")
        return theta.copy(), ridge_solve_diagnostics(parent_model)

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

    def _model_from_solve(self, result: RidgeSolveResult) -> AffineRidgeModel:
        model = AffineRidgeModel(
            ridge=self.ridge,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )
        model.coef_ = result.coefficients.copy()
        model.solve_result_ = result
        return model

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    def _required_side_points(self, n_total: int) -> int:
        fraction_count = int(np.ceil(self.min_side_fraction * n_total))
        return max(int(self.min_side_points), fraction_count)
