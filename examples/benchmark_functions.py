"""Reusable benchmark functions for regression-tree experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


RotationMode = Literal["identity", "random"]
PlaneWaveMode = Literal["explicit", "random"]
RandomState = int | np.random.Generator | None

QUADRATIC_DEFAULT_INTERVAL = (-3.0, 3.0)
GAUSSIAN_DEFAULT_INTERVAL = (-3.0, 3.0)
PLANE_WAVE_DEFAULT_INTERVAL = (-3.0, 3.0)
SPHERICAL_PIECEWISE_DEFAULT_INTERVAL = QUADRATIC_DEFAULT_INTERVAL
ROSENBROCK_DEFAULT_INTERVAL = (-2.0, 3.0)


def rotation_matrix_2d(angle: float, degrees: bool = False) -> np.ndarray:
    """Return the counterclockwise 2D rotation matrix for an angle."""

    angle = float(angle)
    if not np.isfinite(angle):
        raise ValueError("angle must be finite.")
    if degrees:
        angle = float(np.deg2rad(angle))

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ]
    )


def random_orthogonal_matrix(
    dimension: int,
    random_state: RandomState = None,
) -> np.ndarray:
    """Generate a reproducible Haar-distributed orthogonal matrix."""

    dimension = _validate_dimension(dimension, minimum=1)
    rng = _as_rng(random_state)
    Q, R = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
    return Q * signs[None, :]


@dataclass
class QuadraticFunction:
    """Anisotropic quadratic: beta + w^T x + x^T(Q Lambda Q^T)x."""

    dimension: int
    beta: float = 1.0
    w: np.ndarray | None = None
    Q: np.ndarray | None = None
    Lambda: np.ndarray | None = None
    q_mode: RotationMode = "identity"
    random_state: RandomState = None
    quadratic_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.beta = float(self.beta)
        self.w = (
            np.ones(self.dimension, dtype=float)
            if self.w is None
            else _validate_vector(self.w, self.dimension, "w")
        )
        self.Q = _resolve_rotation(
            self.dimension,
            self.Q,
            self.q_mode,
            self.random_state,
        )
        self.Lambda = _resolve_symmetric_matrix(
            self.Lambda,
            self.dimension,
            "Lambda",
            default_diagonal=_paper_diagonal(self.dimension),
        )
        self.quadratic_matrix = self.Q @ self.Lambda @ self.Q.T

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        values = (
            self.beta
            + X @ self.w
            + np.einsum("ni,ij,nj->n", X, self.quadratic_matrix, X)
        )
        return _restore_output(values, scalar_input)


@dataclass
class SphericalPiecewisePolynomialFunction:
    """Use one quadratic polynomial inside a sphere and another outside."""

    inside_polynomial: QuadraticFunction
    outside_polynomial: QuadraticFunction
    radius: float
    center: np.ndarray | None = None
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.inside_polynomial, QuadraticFunction):
            raise TypeError("inside_polynomial must be a QuadraticFunction.")
        if not isinstance(self.outside_polynomial, QuadraticFunction):
            raise TypeError("outside_polynomial must be a QuadraticFunction.")
        if self.inside_polynomial.dimension != self.outside_polynomial.dimension:
            raise ValueError("Inside and outside polynomials must have the same dimension.")

        self.dimension = self.inside_polynomial.dimension
        self.radius = _validate_finite_scalar(self.radius, "radius")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        self.center = (
            np.zeros(self.dimension, dtype=float)
            if self.center is None
            else _validate_vector(self.center, self.dimension, "center")
        )

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        inside = np.sum((X - self.center) ** 2, axis=1) <= self.radius**2
        values = np.empty(X.shape[0], dtype=float)
        if np.any(inside):
            values[inside] = self.inside_polynomial(X[inside])
        if np.any(~inside):
            values[~inside] = self.outside_polynomial(X[~inside])
        return _restore_output(values, scalar_input)


@dataclass
class PlaneWaveFunction:
    """Plane wave beta + amplitude*cos(frequency*normal^T*x + phase)."""

    dimension: int
    beta: float = 0.0
    frequency: float = 1.0
    normal: np.ndarray | None = None
    phase: float | None = None
    amplitude: float = 1.0
    feature_mode: PlaneWaveMode = "explicit"
    random_state: RandomState = None

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.beta = _validate_finite_scalar(self.beta, "beta")
        self.frequency = _validate_finite_scalar(self.frequency, "frequency")
        self.amplitude = _validate_finite_scalar(self.amplitude, "amplitude")
        if self.frequency < 0.0:
            raise ValueError("frequency must be nonnegative.")
        if self.feature_mode not in ("explicit", "random"):
            raise ValueError("feature_mode must be 'explicit' or 'random'.")

        if self.feature_mode == "random":
            if self.normal is not None or self.phase is not None:
                raise ValueError(
                    "normal and phase must be omitted when feature_mode='random'."
                )
            rng = _as_rng(self.random_state)
            self.normal = _random_unit_vector(self.dimension, rng)
            self.phase = float(rng.uniform(0.0, 2.0 * np.pi))
        else:
            if self.normal is None:
                normal = np.zeros(self.dimension, dtype=float)
                normal[0] = 1.0
            else:
                normal = _validate_vector(self.normal, self.dimension, "normal")
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError("normal must be finite and nonzero.")
            self.normal = normal / norm
            self.phase = (
                0.0
                if self.phase is None
                else _validate_finite_scalar(self.phase, "phase")
            )

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        values = self.beta + self.amplitude * np.cos(
            self.frequency * (X @ self.normal) + self.phase
        )
        return _restore_output(values, scalar_input)


@dataclass
class GaussianFunction:
    """Anisotropic Gaussian centered at mean with a cached precision matrix."""

    dimension: int
    beta: float = 1.0
    Q: np.ndarray | None = None
    Sigma: np.ndarray | None = None
    q_mode: RotationMode = "identity"
    random_state: RandomState = None
    mean: np.ndarray | None = None
    covariance: np.ndarray = field(init=False, repr=False)
    precision: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.beta = float(self.beta)
        self.mean = (
            np.zeros(self.dimension, dtype=float)
            if self.mean is None
            else _validate_vector(self.mean, self.dimension, "mean")
        )
        self.Q = _resolve_rotation(
            self.dimension,
            self.Q,
            self.q_mode,
            self.random_state,
        )
        self.Sigma = _resolve_symmetric_matrix(
            self.Sigma,
            self.dimension,
            "Sigma",
            default_diagonal=_paper_diagonal(self.dimension),
        )
        if np.min(np.linalg.eigvalsh(self.Sigma)) <= 0.0:
            raise ValueError("Sigma must be positive definite.")
        self.covariance = self.Q @ self.Sigma @ self.Q.T
        # Cache the inverse once; oracle evaluations only use matrix products.
        self.precision = np.linalg.inv(self.covariance)

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        centered = X - self.mean
        exponent = -0.5 * np.einsum(
            "ni,ij,nj->n",
            centered,
            self.precision,
            centered,
        )
        values = self.beta + np.exp(exponent)
        return _restore_output(values, scalar_input)


@dataclass
class GaussianMixtureFunction:
    """A weighted mixture of zero-bias Gaussian components."""

    components: tuple[GaussianFunction, ...]
    weights: np.ndarray
    beta: float = 0.0
    normalize_weights: bool = True
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        self.components = tuple(self.components)
        if not self.components:
            raise ValueError("components must be nonempty.")
        if not all(
            isinstance(component, GaussianFunction) for component in self.components
        ):
            raise TypeError("components must contain only GaussianFunction instances.")

        self.dimension = self.components[0].dimension
        if any(component.dimension != self.dimension for component in self.components):
            raise ValueError("All Gaussian components must have the same dimension.")
        if any(component.beta != 0.0 for component in self.components):
            raise ValueError("Gaussian mixture components must have beta=0.")

        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if weights.shape[0] != len(self.components):
            raise ValueError("weights must contain one value per component.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and nonnegative.")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("At least one mixture weight must be positive.")
        self.weights = weights / total_weight if self.normalize_weights else weights.copy()
        self.beta = float(self.beta)
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite.")

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        values = np.full(X.shape[0], self.beta, dtype=float)
        for weight, component in zip(self.weights, self.components):
            values += weight * np.asarray(component(X), dtype=float)
        return _restore_output(values, scalar_input)


def default_gaussian_mixture_2d(beta: float = 1.0) -> GaussianMixtureFunction:
    """Return the fixed four-component Gaussian mixture benchmark."""

    centers = (
        (-1.5, -1.1),
        (1.35, -1.2),
        (-1.0, 1.45),
        (1.4, 1.3),
    )
    angles = (20.0, -35.0, 60.0, 5.0)
    spectra = (
        (2.55, 0.125),
        (0.30, 5.00),
        (2.75, 2.18),
        (0.42, 0.10),
    )
    components = tuple(
        GaussianFunction(
            dimension=2,
            beta=0.0,
            Q=rotation_matrix_2d(angle, degrees=True),
            Sigma=np.diag(spectrum),
            mean=np.asarray(center, dtype=float),
        )
        for center, angle, spectrum in zip(centers, angles, spectra)
    )
    return GaussianMixtureFunction(
        components=components,
        weights=np.array([0.0, 1.30, 2.00, 0.0]),
        beta=beta,
        normalize_weights=True,
    )


@dataclass
class RosenbrockFunction:
    """Sum b*(x[j+1] - x[j]^2)^2 + (a - x[j])^2 over adjacent pairs."""

    dimension: int
    a: float = 1.0
    b: float = 100.0

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=2)
        self.a = float(self.a)
        self.b = float(self.b)
        if self.b < 0.0:
            raise ValueError("b must be nonnegative.")

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        values = np.sum(
            self.b * (X[:, 1:] - X[:, :-1] ** 2) ** 2
            + (self.a - X[:, :-1]) ** 2,
            axis=1,
        )
        return _restore_output(values, scalar_input)


def _paper_diagonal(dimension: int) -> np.ndarray:
    indices = np.arange(1, dimension + 1, dtype=float)
    return indices**-3


def _resolve_rotation(
    dimension: int,
    Q: np.ndarray | None,
    mode: RotationMode,
    random_state: RandomState,
) -> np.ndarray:
    if mode not in ("identity", "random"):
        raise ValueError("q_mode must be 'identity' or 'random'.")
    if Q is not None and mode != "identity":
        raise ValueError("Specify either an explicit Q or q_mode='random', not both.")

    if Q is None:
        if mode == "random":
            return random_orthogonal_matrix(dimension, random_state=random_state)
        return np.eye(dimension)

    Q = np.asarray(Q, dtype=float)
    if Q.shape != (dimension, dimension):
        raise ValueError(f"Q must have shape {(dimension, dimension)}, got {Q.shape}.")
    if not np.allclose(Q.T @ Q, np.eye(dimension), atol=1e-10, rtol=1e-10):
        raise ValueError("Q must be orthogonal.")
    return Q.copy()


def _resolve_symmetric_matrix(
    matrix: np.ndarray | None,
    dimension: int,
    name: str,
    default_diagonal: np.ndarray,
) -> np.ndarray:
    if matrix is None:
        return np.diag(default_diagonal)

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        if matrix.shape[0] != dimension:
            raise ValueError(f"{name} diagonal must have length {dimension}.")
        return np.diag(matrix)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape ({dimension},) or {(dimension, dimension)}, "
            f"got {matrix.shape}."
        )
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    return matrix.copy()


def _as_points(x: np.ndarray, dimension: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(x, dtype=float)
    scalar_input = values.ndim == 1
    if scalar_input:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"x must have shape ({dimension},) or (n, {dimension}), got {values.shape}.")
    return values, scalar_input


def _restore_output(values: np.ndarray, scalar_input: bool) -> float | np.ndarray:
    return float(values[0]) if scalar_input else values


def _validate_dimension(dimension: int, minimum: int) -> int:
    dimension = int(dimension)
    if dimension < minimum:
        raise ValueError(f"dimension must be at least {minimum}.")
    return dimension


def _validate_finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_vector(values: np.ndarray, dimension: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != dimension:
        raise ValueError(f"{name} must have shape ({dimension},), got {values.shape}.")
    return values.copy()


def _random_unit_vector(
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    while True:
        vector = rng.normal(size=dimension)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0 and np.isfinite(norm):
            return vector / norm


def _as_rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
