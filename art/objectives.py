"""Objective functions for split optimization."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

from .models import (
    AffineRidgeModel,
    PolynomialRidgeModel,
    PreparedDesign,
    PreparedFeatureModel,
    RegressionModel,
    WeightedRegressionModel,
    augment_features,
    ridge_solve_diagnostics,
)
from .timing import BuildTimingProfile, SplitterTimingProfile

if TYPE_CHECKING:
    from ._cuda_hybrid_gram import CudaWeightedGram


def make_cuda_workspace(
    model: WeightedRegressionModel,
    X: np.ndarray,
    y: np.ndarray,
    prepared_design: PreparedDesign | None,
    weight_floor: float,
) -> CudaWeightedGram:
    """Create a lazy workspace only for the built-in normal-equation models."""
    if type(model) not in (AffineRidgeModel, PolynomialRidgeModel):
        raise TypeError("CUDA Gram construction supports the built-in affine/polynomial models.")
    if model.solver not in ("normal", "auto"):
        raise ValueError("CUDA Gram construction requires solver='normal' or 'auto'.")
    from ._cuda_hybrid_gram import CudaWeightedGram

    if prepared_design is None and isinstance(model, PolynomialRidgeModel):
        prepared_design = model.prepare_design(X)
    design = (
        prepared_design.matrix if prepared_design is not None else augment_features(X)
    )
    return CudaWeightedGram(design, y, weight_floor)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class SoftObjectiveResult:
    """Cached evaluation state for one soft-oblique candidate."""

    theta: np.ndarray
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
    X: np.ndarray
    y: np.ndarray
    weight_floor: float = 1e-12
    prepared_design: PreparedDesign | None = None
    timing_profile: BuildTimingProfile | None = field(
        default=None,
        repr=False,
    )
    split_timing_profile: SplitterTimingProfile | None = field(
        default=None,
        repr=False,
    )
    optimizer: Literal["cpu", "hybrid_gram"] = "cpu"
    gram_workspace: CudaWeightedGram | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
        if self.X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {self.X.shape}.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {self.X.shape[0]} != {self.y.shape[0]}."
            )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not hasattr(self.model_template, "fit_weighted"):
            raise TypeError("model_template must provide fit_weighted(X, y, weights).")
        if (
            self.optimizer == "hybrid_gram" and self.prepared_design is None
            and isinstance(self.model_template, PolynomialRidgeModel)
        ):
            self.prepared_design = self.model_template.prepare_design(self.X)
        if self.prepared_design is not None:
            if not isinstance(self.model_template, PreparedFeatureModel):
                raise TypeError(
                    "prepared_design requires a PreparedFeatureModel template."
                )
            if self.prepared_design.n_samples != self.X.shape[0]:
                raise ValueError(
                    "Prepared design and X must have the same number of rows."
                )
            if self.prepared_design.input_dimension != self.X.shape[1]:
                raise ValueError(
                    "Prepared design input dimension does not match X."
                )
        if self.optimizer not in ("cpu", "hybrid_gram"):
            raise ValueError("The CPU objective supports optimizer='cpu' or 'hybrid_gram'.")
        if self.gram_workspace is not None:
            if self.optimizer != "hybrid_gram":
                raise ValueError("A CUDA Gram workspace requires optimizer='hybrid_gram'.")
            design = (
                self.prepared_design.matrix
                if self.prepared_design is not None else augment_features(self.X)
            )
            self.gram_workspace.validate(design, self.y, self.weight_floor)

    def evaluate(
        self,
        theta: np.ndarray,
        reference: SoftObjectiveResult | None = None,
    ) -> SoftObjectiveResult:
        """Evaluate shared candidate state, optionally reusing fitted models."""

        theta = self._validate_theta(theta)
        w = theta[:-1]
        z = float(theta[-1])
        sigmoid_timing = (
            self.split_timing_profile.measure("sigmoid", count=True)
            if self.split_timing_profile is not None
            else nullcontext()
        )
        with sigmoid_timing:
            pi = sigmoid((self.X @ w - z) / self.temperature)

        if reference is None:
            left_model, right_model = self._fit_models(pi)
            residual_left = self.y - self._predict_model(left_model)
            residual_right = self.y - self._predict_model(right_model)
        else:
            left_model = reference.left_model
            right_model = reference.right_model
            residual_left = reference.residual_left
            residual_right = reference.residual_right
        return SoftObjectiveResult(
            theta=theta.copy(),
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
                "prepared_design": self.prepared_design is not None,
                "left_solve": ridge_solve_diagnostics(left_model),
                "right_solve": ridge_solve_diagnostics(right_model),
            },
        )

    def value(
        self,
        evaluation_or_theta: SoftObjectiveResult | np.ndarray,
    ) -> float:
        """Return a cached loss or evaluate one from theta."""

        evaluation = self._resolve_evaluation(evaluation_or_theta)
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
            timing = (
                self.split_timing_profile.measure("gradient", count=True)
                if self.split_timing_profile is not None
                else nullcontext()
            )
            with timing:
                evaluation.grad = self._gradient(
                    evaluation.pi,
                    evaluation.residual_left,
                    evaluation.residual_right,
                )
        return evaluation.grad

    def value_and_grad(
        self,
        theta: np.ndarray,
        reference: SoftObjectiveResult | None = None,
    ) -> SoftObjectiveResult:
        """Evaluate a candidate and populate both cached loss and gradient."""

        evaluation = self.evaluate(theta, reference=reference)
        self.value(evaluation)
        self.grad(evaluation)
        return evaluation

    def _fit_models(
        self,
        pi: np.ndarray,
    ) -> tuple[RegressionModel, RegressionModel]:
        build_timing = (
            self.timing_profile.measure("model_refit")
            if self.timing_profile is not None
            else nullcontext()
        )
        with build_timing:
            return self._fit_models_untimed(pi)

    def _fit_models_untimed(
        self,
        pi: np.ndarray,
    ) -> tuple[RegressionModel, RegressionModel]:
        left_model = self.model_template.clone()
        right_model = self.model_template.clone()
        solve_timing = (
            self.split_timing_profile.measure("least_squares", count=True)
            if self.split_timing_profile is not None
            else nullcontext()
        )
        with solve_timing:
            if self.optimizer == "hybrid_gram":
                if self.gram_workspace is None:
                    self.gram_workspace = make_cuda_workspace(
                        self.model_template, self.X, self.y,
                        self.prepared_design, self.weight_floor,
                    )
                systems = self.gram_workspace.form_pair(pi)
                for model, weights, system in zip(
                    (left_model, right_model), (1.0 - pi, pi), systems
                ):
                    if self.prepared_design is None:
                        model.fit_weighted(
                            self.X, self.y, weights,
                            weight_floor=self.weight_floor,
                            normal_equations=system,
                        )
                    else:
                        model.fit_weighted_design(
                            self.prepared_design, self.y, weights,
                            weight_floor=self.weight_floor,
                            normal_equations=system,
                        )
            elif self.prepared_design is None:
                left_model.fit_weighted(
                    self.X,
                    self.y,
                    weights=1.0 - pi,
                    weight_floor=self.weight_floor,
                )
                right_model.fit_weighted(
                    self.X,
                    self.y,
                    weights=pi,
                    weight_floor=self.weight_floor,
                )
            else:
                if not isinstance(left_model, PreparedFeatureModel) or not isinstance(
                    right_model, PreparedFeatureModel
                ):
                    raise TypeError(
                        "Prepared objective models must implement PreparedFeatureModel."
                    )
                left_model.fit_weighted_design(
                    self.prepared_design,
                    self.y,
                    weights=1.0 - pi,
                    weight_floor=self.weight_floor,
                )
                right_model.fit_weighted_design(
                    self.prepared_design,
                    self.y,
                    weights=pi,
                    weight_floor=self.weight_floor,
                )
        return left_model, right_model

    def _predict_model(self, model: RegressionModel) -> np.ndarray:
        if self.prepared_design is None:
            return model.predict(self.X)
        if not isinstance(model, PreparedFeatureModel):
            raise TypeError(
                "Prepared objective models must implement PreparedFeatureModel."
            )
        return model.predict_design(self.prepared_design)

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
        pi: np.ndarray,
        residual_left: np.ndarray,
        residual_right: np.ndarray,
    ) -> np.ndarray:
        n = self.X.shape[0]
        coeff = (
            (residual_right**2 - residual_left**2)
            * pi
            * (1.0 - pi)
            / n
        )
        grad_w = self.X.T @ coeff / self.temperature
        grad_z = -float(np.sum(coeff)) / self.temperature
        return np.concatenate([grad_w, np.array([grad_z])])

    def _resolve_evaluation(
        self,
        evaluation_or_theta: SoftObjectiveResult | np.ndarray,
    ) -> SoftObjectiveResult:
        if isinstance(evaluation_or_theta, SoftObjectiveResult):
            return evaluation_or_theta
        return self.evaluate(evaluation_or_theta)

    def _validate_theta(
        self,
        theta: np.ndarray,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape[0] != self.X.shape[1] + 1:
            raise ValueError(
                f"theta must have length {self.X.shape[1] + 1}, "
                f"got {theta.shape[0]}."
            )
        return theta
