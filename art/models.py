"""Regression model interfaces and basic implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Literal, Protocol, runtime_checkable

import numpy as np


RidgeSolver = Literal["auto", "normal", "qr", "svd"]
ResolvedRidgeSolver = Literal["cholesky", "qr", "svd"]
ConditionEstimator = Literal[
    "sqrt_gram_1norm",
    "svd_2norm",
    "cholesky_failure",
    "condition_estimate_failure",
]
RIDGE_SOLVERS: tuple[RidgeSolver, ...] = ("auto", "normal", "qr", "svd")
DEFAULT_AUTO_RCOND_THRESHOLD = 1e-10
MODEL_CONDITION_WARNING_THRESHOLD = 1e10


@dataclass(frozen=True)
class RidgeSolveResult:
    """Coefficients and numerical diagnostics from one ridge solve."""

    coefficients: np.ndarray
    solver_requested: RidgeSolver
    solver_used: ResolvedRidgeSolver
    condition_estimator: ConditionEstimator | None
    cond_estimate: float | None
    ridge_effective: float
    rank: int | None = None
    fallback_reason: str | None = None


@runtime_checkable
class RegressionModel(Protocol):
    """Protocol expected by the tree builder and split optimizers."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def clone(self) -> "RegressionModel":
        ...


@runtime_checkable
class WeightedRegressionModel(RegressionModel, Protocol):
    """Regression model that supports weighted least-squares style fitting."""

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "WeightedRegressionModel":
        ...


@dataclass(frozen=True)
class PreparedDesign:
    """Feature matrix and fitted transform state for one ordered sample set."""

    matrix: np.ndarray
    input_dimension: int
    feature_signature: tuple[object, ...]
    feature_state: object

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(
                f"Prepared design matrix must have shape (n, p), got {matrix.shape}."
            )
        if self.input_dimension < 1:
            raise ValueError("Prepared design input_dimension must be at least 1.")
        object.__setattr__(self, "matrix", matrix)

    @property
    def n_samples(self) -> int:
        return self.matrix.shape[0]

    def subset(self, rows: np.ndarray) -> "PreparedDesign":
        """Select rows without recomputing the feature transformation."""

        matrix = self.matrix[rows]
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return PreparedDesign(
            matrix=matrix,
            input_dimension=self.input_dimension,
            feature_signature=self.feature_signature,
            feature_state=self.feature_state,
        )


@runtime_checkable
class PreparedFeatureModel(WeightedRegressionModel, Protocol):
    """Weighted model supporting reusable, explicitly prepared features."""

    def prepare_design(self, X: np.ndarray) -> PreparedDesign:
        ...

    def fit_design(
        self,
        prepared: PreparedDesign,
        y: np.ndarray,
    ) -> "PreparedFeatureModel":
        ...

    def fit_weighted_design(
        self,
        prepared: PreparedDesign,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "PreparedFeatureModel":
        ...

    def predict_design(self, prepared: PreparedDesign) -> np.ndarray:
        ...


def polynomial_feature_count(
    input_dimension: int,
    degree: int,
    include_bias: bool = True,
) -> int:
    """Return the number of monomials up to the requested degree."""

    input_dimension = int(input_dimension)
    degree = int(degree)
    if input_dimension < 1:
        raise ValueError("input_dimension must be at least 1.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    count = comb(input_dimension + degree, degree)
    return count if include_bias else count - 1


def model_effective_dimension(model: RegressionModel, input_dimension: int) -> int:
    """Return a model's finite fitting dimension for sample budgeting."""

    method = getattr(model, "effective_dimension", None)
    if method is None:
        raise TypeError(
            f"{type(model).__name__} does not define effective_dimension; "
            "provide an explicit sample_count policy to the tree builder."
        )
    dimension = int(method(input_dimension))
    if dimension < 1:
        raise ValueError("Model effective_dimension must be at least 1.")
    return dimension


def ridge_solve_diagnostics(model: object) -> dict[str, object] | None:
    """Return scalar diagnostics from a model or ridge-solve result."""

    result = model if isinstance(model, RidgeSolveResult) else getattr(
        model, "solve_result_", None
    )
    if not isinstance(result, RidgeSolveResult):
        return None
    return {
        "solver_requested": result.solver_requested,
        "solver_used": result.solver_used,
        "condition_estimator": result.condition_estimator,
        "cond_estimate": result.cond_estimate,
        "ridge_effective": result.ridge_effective,
        "rank": result.rank,
        "fallback_reason": result.fallback_reason,
    }


def augment_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    return np.hstack([X, np.ones((X.shape[0], 1))])


def scaled_ridge_from_gram(gram: np.ndarray, ridge: float) -> float:
    """Scale ridge by the mean eigenvalue of the Gram matrix."""

    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square, got {gram.shape}.")
    return float(ridge) * float(np.trace(gram)) / gram.shape[0]


def solve_weighted_ridge(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    weight_floor: float = 1e-12,
    solver: RidgeSolver = "normal",
    auto_rcond_threshold: float = DEFAULT_AUTO_RCOND_THRESHOLD,
) -> RidgeSolveResult:
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if design.ndim != 2:
        raise ValueError(f"design must have shape (n, p), got {design.shape}.")
    if design.shape[0] != y.shape[0] or y.shape[0] != weights.shape[0]:
        raise ValueError("design, y, and weights must have the same number of rows.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative.")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative.")
    _validate_ridge_solver(solver)
    _validate_auto_rcond_threshold(auto_rcond_threshold)

    safe_weights = np.maximum(weights, float(weight_floor))
    from scipy import linalg

    if solver in ("auto", "normal"):
        gram = design.T @ (design * safe_weights[:, None])
        rhs = design.T @ (y * safe_weights)
        ridge_eff = scaled_ridge_from_gram(gram, ridge)
        regularized_gram = gram.copy()
        regularized_gram.flat[:: gram.shape[0] + 1] += ridge_eff
        gram_norm = linalg.norm(regularized_gram, 1, check_finite=False)
        try:
            factor = linalg.cho_factor(
                regularized_gram,
                lower=True,
                overwrite_a=True,
                check_finite=False,
            )
        except linalg.LinAlgError as exc:
            if solver == "normal":
                raise np.linalg.LinAlgError(
                    "Normal-equation Cholesky factorization failed; use "
                    "solver='auto', solver='qr', or solver='svd', or increase "
                    "ridge regularization."
                ) from exc
            return _solve_augmented_ridge(
                design,
                y,
                safe_weights,
                ridge_eff,
                solver_requested=solver,
                solver_used="qr",
                condition_estimator="cholesky_failure",
                cond_estimate=np.inf,
                fallback_reason="cholesky_failed",
            )

        from scipy.linalg import lapack

        gram_rcond, info = lapack.dpocon(factor[0], gram_norm, uplo="L")
        if info == 0 and gram_rcond > 0.0:
            cond_estimate = float(np.sqrt(1.0 / min(float(gram_rcond), 1.0)))
            condition_estimator: ConditionEstimator = "sqrt_gram_1norm"
        else:
            gram_rcond = 0.0
            cond_estimate = np.inf
            condition_estimator = "condition_estimate_failure"

        if solver == "normal" or gram_rcond >= auto_rcond_threshold:
            coefficients = linalg.cho_solve(factor, rhs, check_finite=False)
            return RidgeSolveResult(
                coefficients=coefficients,
                solver_requested=solver,
                solver_used="cholesky",
                condition_estimator=condition_estimator,
                cond_estimate=cond_estimate,
                ridge_effective=ridge_eff,
            )

        return _solve_augmented_ridge(
            design,
            y,
            safe_weights,
            ridge_eff,
            solver_requested=solver,
            solver_used="qr",
            condition_estimator=condition_estimator,
            cond_estimate=cond_estimate,
            fallback_reason="rcond_below_threshold",
        )

    sqrt_weights = np.sqrt(safe_weights)
    weighted_design = design * sqrt_weights[:, None]
    ridge_eff = float(ridge) * float(np.sum(weighted_design**2)) / design.shape[1]
    return _solve_augmented_ridge(
        design,
        y,
        safe_weights,
        ridge_eff,
        solver_requested=solver,
        solver_used=solver,
    )


def _solve_augmented_ridge(
    design: np.ndarray,
    y: np.ndarray,
    safe_weights: np.ndarray,
    ridge_eff: float,
    solver_requested: RidgeSolver,
    solver_used: Literal["qr", "svd"],
    condition_estimator: ConditionEstimator | None = None,
    cond_estimate: float | None = None,
    fallback_reason: str | None = None,
) -> RidgeSolveResult:
    from scipy import linalg

    sqrt_weights = np.sqrt(safe_weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = y * sqrt_weights
    if ridge_eff > 0.0:
        augmented_design = np.vstack(
            [weighted_design, np.sqrt(ridge_eff) * np.eye(design.shape[1])]
        )
        augmented_target = np.concatenate(
            [weighted_target, np.zeros(design.shape[1])]
        )
    else:
        augmented_design = weighted_design
        augmented_target = weighted_target

    driver = "gelsy" if solver_used == "qr" else "gelsd"
    try:
        coefficients, _, rank, singular_values = linalg.lstsq(
            augmented_design,
            augmented_target,
            lapack_driver=driver,
            check_finite=False,
        )
    except linalg.LinAlgError:
        if solver_requested != "auto" or solver_used != "qr":
            raise
        return _solve_augmented_ridge(
            design,
            y,
            safe_weights,
            ridge_eff,
            solver_requested=solver_requested,
            solver_used="svd",
            fallback_reason="qr_failed",
        )

    if solver_requested == "auto" and solver_used == "qr" and not np.all(
        np.isfinite(coefficients)
    ):
        return _solve_augmented_ridge(
            design,
            y,
            safe_weights,
            ridge_eff,
            solver_requested=solver_requested,
            solver_used="svd",
            fallback_reason="qr_nonfinite",
        )

    if solver_used == "svd":
        condition_estimator = "svd_2norm"
        cond_estimate = _condition_from_singular_values(singular_values)

    return RidgeSolveResult(
        coefficients=coefficients,
        solver_requested=solver_requested,
        solver_used=solver_used,
        condition_estimator=condition_estimator,
        cond_estimate=cond_estimate,
        ridge_effective=ridge_eff,
        rank=int(rank),
        fallback_reason=fallback_reason,
    )


def _condition_from_singular_values(
    singular_values: np.ndarray | None,
) -> float | None:
    if singular_values is None or singular_values.size == 0:
        return None
    smallest = float(singular_values[-1])
    return np.inf if smallest <= 0.0 else float(singular_values[0] / smallest)


def _validate_ridge_solver(solver: str) -> None:
    if solver not in RIDGE_SOLVERS:
        raise ValueError(
            f"solver must be one of {RIDGE_SOLVERS}, got {solver!r}."
        )


def _validate_auto_rcond_threshold(value: float) -> None:
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError("auto_rcond_threshold must satisfy 0 < value <= 1.")


@dataclass
class AffineRidgeModel:
    """Affine model fit by ridge-regularized least squares."""

    ridge: float = 1e-8
    solver: RidgeSolver = "normal"
    auto_rcond_threshold: float = DEFAULT_AUTO_RCOND_THRESHOLD
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    solve_result_: RidgeSolveResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Load the solver dependency when the model template is constructed so
        # its one-time import cost is not charged to the first timed fit.
        import scipy.linalg  # noqa: F401

        _validate_ridge_solver(self.solver)
        _validate_auto_rcond_threshold(self.auto_rcond_threshold)

    def effective_dimension(self, input_dimension: int) -> int:
        input_dimension = int(input_dimension)
        if input_dimension < 1:
            raise ValueError("input_dimension must be at least 1.")
        return input_dimension + 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AffineRidgeModel":
        X_aug = augment_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X_aug.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X_aug.shape[0]} != {y.shape[0]}.")

        self.solve_result_ = solve_weighted_ridge(
            X_aug,
            y,
            np.ones_like(y),
            self.ridge,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )
        self.coef_ = self.solve_result_.coefficients
        return self

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "AffineRidgeModel":
        self.solve_result_ = solve_weighted_ridge(
            augment_features(X),
            y,
            weights,
            ridge=self.ridge,
            weight_floor=weight_floor,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )
        self.coef_ = self.solve_result_.coefficients
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        if X.shape[1] != self.coef_.shape[0] - 1:
            raise ValueError(
                f"X must have {self.coef_.shape[0] - 1} columns, got {X.shape[1]}."
            )
        return X @ self.coef_[:-1] + self.coef_[-1]

    def clone(self) -> "AffineRidgeModel":
        return AffineRidgeModel(
            ridge=self.ridge,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )


@dataclass
class PolynomialRidgeModel:
    """Polynomial feature model fit by ridge-regularized least squares."""

    degree: int = 2
    ridge: float = 1e-8
    include_bias: bool = True
    solver: RidgeSolver = "normal"
    auto_rcond_threshold: float = DEFAULT_AUTO_RCOND_THRESHOLD
    transformer_: object | None = field(default=None, init=False, repr=False)
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    solve_result_: RidgeSolveResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from sklearn.preprocessing import PolynomialFeatures

        _validate_ridge_solver(self.solver)
        _validate_auto_rcond_threshold(self.auto_rcond_threshold)
        self.transformer_ = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)

    def effective_dimension(self, input_dimension: int) -> int:
        return polynomial_feature_count(
            input_dimension,
            self.degree,
            include_bias=self.include_bias,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRidgeModel":
        y = np.asarray(y, dtype=float).reshape(-1)
        return self.fit_design(self.prepare_design(X), y)

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "PolynomialRidgeModel":
        return self.fit_weighted_design(
            self.prepare_design(X),
            y,
            weights,
            weight_floor=weight_floor,
        )

    def prepare_design(self, X: np.ndarray) -> PreparedDesign:
        from sklearn.preprocessing import PolynomialFeatures

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")

        transformer = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
        )
        return PreparedDesign(
            matrix=transformer.fit_transform(X),
            input_dimension=X.shape[1],
            feature_signature=self._feature_signature(),
            feature_state=transformer,
        )

    def fit_design(
        self,
        prepared: PreparedDesign,
        y: np.ndarray,
    ) -> "PolynomialRidgeModel":
        y = np.asarray(y, dtype=float).reshape(-1)
        return self.fit_weighted_design(prepared, y, np.ones_like(y))

    def fit_weighted_design(
        self,
        prepared: PreparedDesign,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "PolynomialRidgeModel":
        self._validate_prepared(prepared)
        self.solve_result_ = solve_weighted_ridge(
            prepared.matrix,
            y,
            weights,
            ridge=self.ridge,
            weight_floor=weight_floor,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )
        self.transformer_ = prepared.feature_state
        self.coef_ = self.solve_result_.coefficients
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict.")
        design = self.transformer_.transform(np.asarray(X, dtype=float))
        return design @ self.coef_

    def predict_design(self, prepared: PreparedDesign) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict_design.")
        self._validate_prepared(prepared)
        if prepared.matrix.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "Prepared design feature count does not match fitted coefficients."
            )
        return prepared.matrix @ self.coef_

    def clone(self) -> "PolynomialRidgeModel":
        return PolynomialRidgeModel(
            degree=self.degree,
            ridge=self.ridge,
            include_bias=self.include_bias,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )

    def _feature_signature(self) -> tuple[object, ...]:
        return ("polynomial", self.degree, self.include_bias)

    def _validate_prepared(self, prepared: PreparedDesign) -> None:
        if not isinstance(prepared, PreparedDesign):
            raise TypeError("prepared must be a PreparedDesign.")
        if prepared.feature_signature != self._feature_signature():
            raise ValueError(
                "Prepared design does not match this polynomial model configuration."
            )
        expected_features = self.effective_dimension(prepared.input_dimension)
        if prepared.matrix.shape[1] != expected_features:
            raise ValueError(
                f"Prepared design must have {expected_features} features, "
                f"got {prepared.matrix.shape[1]}."
            )


@dataclass
class KernelRidgeModel:
    """Kernel ridge regression model backed by scikit-learn."""

    # alpha: ridge regularization strength; larger values give smoother fits.
    alpha: float = 1e-6
    # kernel: similarity function, e.g. "rbf", "poly", "linear", or "laplacian".
    kernel: str = "rbf"
    # gamma: inverse length-scale for rbf/laplacian/poly kernels; larger is more local.
    gamma: float | None = None
    # degree: polynomial degree, used when kernel="poly".
    degree: int = 2
    # coef0: offset term for polynomial/sigmoid kernels; controls lower-order terms.
    coef0: float = 1.0
    estimator_: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        self.estimator_ = KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeModel":
        y = np.asarray(y, dtype=float).reshape(-1)
        self.estimator_.fit(np.asarray(X, dtype=float), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator_.predict(np.asarray(X, dtype=float)), dtype=float)

    def clone(self) -> "KernelRidgeModel":
        return KernelRidgeModel(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
