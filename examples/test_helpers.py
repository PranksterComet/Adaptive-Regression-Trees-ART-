"""Shared helpers for synthetic splitter examples."""

from __future__ import annotations

import numpy as np

from art.models import (
    AffineRidgeModel,
    PolynomialRidgeModel,
    augment_features,
    polynomial_feature_count,
)
from art.sampling import sample_uniform_box


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    if any(value <= 0.0 for value in values):
        raise ValueError("All values must be positive.")
    return values


def parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def random_unit_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.normal(size=d)
    norm = np.linalg.norm(w)
    if norm <= 1e-12:
        w[0] = 1.0
        norm = 1.0
    return w / norm


def random_affine_theta(d: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.normal(size=d + 1)
    theta[:-1] /= np.sqrt(d)
    return theta


def random_polynomial_theta(
    d: int,
    degree: int,
    rng: np.random.Generator,
    include_bias: bool = True,
) -> np.ndarray:
    theta = rng.normal(size=polynomial_feature_count(d, degree, include_bias=include_bias))
    return theta / np.sqrt(theta.size)


def make_model_template(degree: int, ridge: float, include_bias: bool = True):
    if degree == 1:
        return AffineRidgeModel(ridge=ridge)
    return PolynomialRidgeModel(degree=degree, ridge=ridge, include_bias=include_bias)


def affine_values(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return augment_features(X) @ theta


def polynomial_values(
    X: np.ndarray,
    theta: np.ndarray,
    degree: int,
    include_bias: bool = True,
) -> np.ndarray:
    from sklearn.preprocessing import PolynomialFeatures

    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        return np.empty(0, dtype=float)
    transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)
    design = transformer.fit_transform(X)
    return design @ theta


def piecewise_affine(
    X: np.ndarray,
    true_w: np.ndarray,
    true_z: float,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
) -> np.ndarray:
    right = (X @ true_w - true_z) >= 0.0
    y = np.empty(X.shape[0], dtype=float)
    y[~right] = affine_values(X[~right], theta_left)
    y[right] = affine_values(X[right], theta_right)
    return y


def piecewise_polynomial(
    X: np.ndarray,
    true_w: np.ndarray,
    true_z: float,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    degree: int,
    include_bias: bool = True,
) -> np.ndarray:
    right = (X @ true_w - true_z) >= 0.0
    y = np.empty(X.shape[0], dtype=float)
    y[~right] = polynomial_values(X[~right], theta_left, degree, include_bias=include_bias)
    y[right] = polynomial_values(X[right], theta_right, degree, include_bias=include_bias)
    return y


def make_piecewise_target(
    true_w: np.ndarray,
    true_z: float,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    degree: int,
    include_bias: bool = True,
):
    def target(X: np.ndarray) -> np.ndarray:
        if degree == 1:
            return piecewise_affine(X, true_w, true_z, theta_left, theta_right)
        return piecewise_polynomial(
            X,
            true_w,
            true_z,
            theta_left,
            theta_right,
            degree=degree,
            include_bias=include_bias,
        )

    return target


def hard_split_predict(X: np.ndarray, result) -> np.ndarray:
    if hasattr(result, "predict"):
        return result.predict(X)
    right = (X @ result.w - result.z) >= 0.0
    y_pred = np.empty(X.shape[0], dtype=float)
    y_pred[~right] = result.left_model.predict(X[~right])
    y_pred[right] = result.right_model.predict(X[right])
    return y_pred


def boundary_misclassification(
    X: np.ndarray,
    true_w: np.ndarray,
    true_z: float,
    learned_w: np.ndarray,
    learned_z: float,
) -> tuple[int, float]:
    true_side = (X @ true_w - true_z) >= 0.0
    learned_side = (X @ learned_w - learned_z) >= 0.0
    n_errors = int(np.sum(true_side != learned_side))
    n_errors_flipped = int(np.sum(true_side != ~learned_side))
    return (n_errors, 1.0) if n_errors <= n_errors_flipped else (n_errors_flipped, -1.0)


def boundary_misclassification_fraction(
    X: np.ndarray,
    true_w: np.ndarray,
    true_z: float,
    learned_w: np.ndarray,
    learned_z: float,
) -> tuple[float, float]:
    n_errors, sign = boundary_misclassification(X, true_w, true_z, learned_w, learned_z)
    return float(n_errors / X.shape[0]), sign


def boundary_errors(
    true_w: np.ndarray,
    true_z: float,
    learned_w: np.ndarray,
    learned_z: float,
    sign: float,
) -> tuple[float, float]:
    aligned_w = sign * learned_w
    aligned_z = sign * learned_z
    dot = float(np.clip(true_w @ aligned_w, -1.0, 1.0))
    angle_degrees = float(np.degrees(np.arccos(dot)))
    offset_error = float(abs(true_z - aligned_z))
    return angle_degrees, offset_error


def sample_balanced_boundary(
    bounds: np.ndarray,
    rng: np.random.Generator,
    min_volume_fraction: float,
    max_attempts: int,
    probe_size: int,
) -> tuple[np.ndarray, float, float]:
    d = bounds.shape[0]
    for _ in range(max_attempts):
        w = random_unit_vector(d, rng)
        z = float(rng.uniform(-0.5, 0.5))
        X_probe = sample_uniform_box(bounds, probe_size, random_state=rng)
        frac_right = float(np.mean((X_probe @ w - z) >= 0.0))
        if min(frac_right, 1.0 - frac_right) >= min_volume_fraction:
            return w, z, frac_right
    raise RuntimeError("Could not sample a sufficiently balanced boundary.")
