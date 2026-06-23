"""Regression model interfaces and basic implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


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
) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if design.ndim != 2:
        raise ValueError(f"design must have shape (n, p), got {design.shape}.")
    if design.shape[0] != y.shape[0] or y.shape[0] != weights.shape[0]:
        raise ValueError("design, y, and weights must have the same number of rows.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative.")

    safe_weights = np.maximum(weights, float(weight_floor))
    gram = design.T @ (design * safe_weights[:, None])
    rhs = design.T @ (y * safe_weights)
    ridge_eff = scaled_ridge_from_gram(gram, ridge)
    return np.linalg.solve(gram + ridge_eff * np.eye(gram.shape[0]), rhs)


@dataclass
class AffineRidgeModel:
    """Affine model fit by ridge-regularized least squares."""

    ridge: float = 1e-8
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AffineRidgeModel":
        X_aug = augment_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X_aug.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X_aug.shape[0]} != {y.shape[0]}.")

        self.coef_ = solve_weighted_ridge(X_aug, y, np.ones_like(y), self.ridge)
        return self

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "AffineRidgeModel":
        self.coef_ = solve_weighted_ridge(
            augment_features(X),
            y,
            weights,
            ridge=self.ridge,
            weight_floor=weight_floor,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict.")
        return augment_features(X) @ self.coef_

    def clone(self) -> "AffineRidgeModel":
        return AffineRidgeModel(ridge=self.ridge)


@dataclass
class PolynomialRidgeModel:
    """Polynomial feature model fit by ridge-regularized least squares."""

    degree: int = 2
    ridge: float = 1e-8
    include_bias: bool = True
    estimator_: object | None = field(default=None, init=False, repr=False)
    transformer_: PolynomialFeatures | None = field(default=None, init=False, repr=False)
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.transformer_ = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.estimator_ = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=self.include_bias),
            Ridge(alpha=self.ridge, fit_intercept=False),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRidgeModel":
        y = np.asarray(y, dtype=float).reshape(-1)
        self.estimator_.fit(np.asarray(X, dtype=float), y)
        self.coef_ = None
        return self

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "PolynomialRidgeModel":
        design = self.transformer_.fit_transform(np.asarray(X, dtype=float))
        self.coef_ = solve_weighted_ridge(
            design,
            y,
            weights,
            ridge=self.ridge,
            weight_floor=weight_floor,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is not None:
            design = self.transformer_.transform(np.asarray(X, dtype=float))
            return design @ self.coef_
        return np.asarray(self.estimator_.predict(np.asarray(X, dtype=float)), dtype=float)

    def clone(self) -> "PolynomialRidgeModel":
        return PolynomialRidgeModel(
            degree=self.degree,
            ridge=self.ridge,
            include_bias=self.include_bias,
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
    estimator_: KernelRidge | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
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
