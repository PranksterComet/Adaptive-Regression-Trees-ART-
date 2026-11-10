"""Reusable benchmark functions for regression-tree experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


RotationMode = Literal["identity", "random"]
PlaneWaveMode = Literal["explicit", "random"]
RandomState = int | np.random.Generator | None

QUADRATIC_DEFAULT_INTERVAL = (-3.0, 3.0)
GAUSSIAN_DEFAULT_INTERVAL = (-1.0, 1.0)
GAUSSIAN_MIXTURE_DEFAULT_INTERVAL = (-3.0, 3.0)
GAUSSIAN_DEFAULT_CONDITION_NUMBER = 16.0
PLANE_WAVE_DEFAULT_INTERVAL = (-3.0, 3.0)
SPHERICAL_PIECEWISE_DEFAULT_INTERVAL = QUADRATIC_DEFAULT_INTERVAL
ROSENBROCK_DEFAULT_INTERVAL = (-2.0, 3.0)
RASTRIGIN_DEFAULT_INTERVAL = (-5.12, 5.12)

FIXED_SPHERICAL_AFFINE_INSIDE_W = (-0.25, 0.35, -0.15, 0.20, -0.10)
FIXED_SPHERICAL_AFFINE_OUTSIDE_W = (0.40, -0.30, 0.20, -0.10, 0.05)
FIXED_SPHERICAL_AFFINE_INSIDE_BETA = 5.0
FIXED_SPHERICAL_AFFINE_OUTSIDE_BETA = 0.0
FIXED_SPHERICAL_AFFINE_OFFSET = 4.15
FIXED_SPHERICAL_AFFINE_VOLUME_FRACTION = 0.5
FIXED_SPHERICAL_AFFINE_RADII = {
    2: 2.394139996923,
    3: 2.954149913692,
    4: 3.418315145003,
    5: 3.834703297290,
}


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


def fixed_spherical_affine_pieces(
    dimension: int,
) -> tuple[QuadraticFunction, QuadraticFunction]:
    """Return the fixed inside and outside affine pieces for dimensions 2--5."""

    dimension = _validate_dimension(dimension, minimum=2)
    max_dimension = len(FIXED_SPHERICAL_AFFINE_INSIDE_W)
    if dimension > max_dimension:
        raise ValueError(
            f"fixed spherical affine pieces support dimension <= {max_dimension}."
        )

    zero_quadratic = np.zeros(dimension, dtype=float)
    inside = QuadraticFunction(
        dimension=dimension,
        beta=FIXED_SPHERICAL_AFFINE_INSIDE_BETA,
        w=np.asarray(FIXED_SPHERICAL_AFFINE_INSIDE_W[:dimension]),
        Lambda=zero_quadratic,
    )
    outside = QuadraticFunction(
        dimension=dimension,
        beta=FIXED_SPHERICAL_AFFINE_OUTSIDE_BETA,
        w=np.asarray(FIXED_SPHERICAL_AFFINE_OUTSIDE_W[:dimension]),
        Lambda=zero_quadratic,
    )
    return inside, outside


def fixed_spherical_affine_radius(dimension: int) -> float:
    """Return the recorded 50%-volume sphere radius for dimensions 2--5."""

    dimension = _validate_dimension(dimension, minimum=2)
    try:
        return FIXED_SPHERICAL_AFFINE_RADII[dimension]
    except KeyError as error:
        raise ValueError(
            "fixed spherical affine radii support dimensions 2--5."
        ) from error


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
    """Scaled anisotropic Gaussian with a cached precision matrix."""

    dimension: int
    beta: float = 1.0
    amplitude: float = 1.0
    sharpness: float = 1.0
    domain_scale: float = 1.0
    condition_number: float = GAUSSIAN_DEFAULT_CONDITION_NUMBER
    Q: np.ndarray | None = None
    Sigma: np.ndarray | None = None
    q_mode: RotationMode = "identity"
    random_state: RandomState = None
    mean: np.ndarray | None = None
    covariance: np.ndarray = field(init=False, repr=False)
    precision: np.ndarray = field(init=False, repr=False)
    precision_spectrum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.beta = _validate_finite_scalar(self.beta, "beta")
        self.amplitude = _validate_finite_scalar(self.amplitude, "amplitude")
        self.sharpness = _validate_finite_scalar(self.sharpness, "sharpness")
        self.domain_scale = _validate_finite_scalar(
            self.domain_scale,
            "domain_scale",
        )
        self.condition_number = _validate_finite_scalar(
            self.condition_number,
            "condition_number",
        )
        if self.sharpness <= 0.0:
            raise ValueError("sharpness must be positive.")
        if self.domain_scale <= 0.0:
            raise ValueError("domain_scale must be positive.")
        if self.condition_number < 1.0:
            raise ValueError("condition_number must be at least 1.")
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
        if self.Sigma is None:
            self.precision_spectrum = _geometric_precision_spectrum(
                self.dimension,
                self.condition_number,
            )
            self.Sigma = np.diag(1.0 / self.precision_spectrum)
            self.precision = (
                self.Q @ np.diag(self.precision_spectrum) @ self.Q.T
            )
            self.covariance = self.Q @ self.Sigma @ self.Q.T
        else:
            self.Sigma = _resolve_symmetric_matrix(
                self.Sigma,
                self.dimension,
                "Sigma",
                default_diagonal=np.ones(self.dimension),
            )
            if np.min(np.linalg.eigvalsh(self.Sigma)) <= 0.0:
                raise ValueError("Sigma must be positive definite.")
            self.covariance = self.Q @ self.Sigma @ self.Q.T
            # Explicit covariance input retains its original interpretation.
            self.precision = np.linalg.inv(self.covariance)
            self.precision_spectrum = np.linalg.eigvalsh(self.precision)[::-1]
        self.condition_number = float(
            self.precision_spectrum[0] / self.precision_spectrum[-1]
        )

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        centered = (X - self.mean) / self.domain_scale
        exponent = -0.5 * self.sharpness * np.einsum(
            "ni,ij,nj->n",
            centered,
            self.precision,
            centered,
        )
        values = self.beta + self.amplitude * np.exp(exponent)
        return _restore_output(values, scalar_input)


@dataclass
class ActiveSubspaceGaussianFunction:
    """Gaussian with strong variation in a seeded low-dimensional subspace."""

    dimension: int
    active_dimension: int = 3
    eta: float = 0.1
    a: float = 1.0
    tau: np.ndarray | None = None
    active_mean: np.ndarray | None = None
    U: np.ndarray | None = None
    random_state: RandomState = None

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.active_dimension = _validate_dimension(
            self.active_dimension,
            minimum=1,
        )
        if self.active_dimension > self.dimension:
            raise ValueError("active_dimension cannot exceed dimension.")

        self.a = _validate_finite_scalar(self.a, "a")
        self.eta = _validate_finite_scalar(self.eta, "eta")
        if self.a <= 0.0:
            raise ValueError("a must be positive.")
        if self.eta < 0.0 or (
            self.active_dimension < self.dimension and self.eta == 0.0
        ):
            raise ValueError(
                "eta must be positive when inactive dimensions are present."
            )

        self.tau = (
            0.2 + 0.3 * np.arange(self.active_dimension, dtype=float)
            if self.tau is None
            else _validate_vector(self.tau, self.active_dimension, "tau")
        )
        if not np.all(np.isfinite(self.tau)) or np.any(self.tau <= 0.0):
            raise ValueError("tau must contain finite positive values.")
        self.active_mean = (
            np.zeros(self.active_dimension, dtype=float)
            if self.active_mean is None
            else _validate_vector(
                self.active_mean,
                self.active_dimension,
                "active_mean",
            )
        )
        if not np.all(np.isfinite(self.active_mean)):
            raise ValueError("active_mean must contain finite values.")

        if self.U is None:
            self.U = random_orthogonal_matrix(
                self.dimension,
                random_state=self.random_state,
            )[:, : self.active_dimension]
        else:
            self.U = np.asarray(self.U, dtype=float)
            expected_shape = (self.dimension, self.active_dimension)
            if self.U.shape != expected_shape:
                raise ValueError(
                    f"U must have shape {expected_shape}, got {self.U.shape}."
                )
            if not np.all(np.isfinite(self.U)):
                raise ValueError("U must contain finite values.")
            if not np.allclose(
                self.U.T @ self.U,
                np.eye(self.active_dimension),
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("U must have orthonormal columns.")
            self.U = self.U.copy()

    @property
    def inactive_variance(self) -> float | None:
        """Return the scaled-coordinate inactive variance, if it exists."""

        inactive_dimension = self.dimension - self.active_dimension
        if inactive_dimension == 0:
            return None
        return inactive_dimension / self.eta

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        scaled = X / self.a
        active_coordinates = scaled @ self.U
        active_energy = np.sum(
            ((active_coordinates - self.active_mean) / self.tau) ** 2,
            axis=1,
        )

        inactive_dimension = self.dimension - self.active_dimension
        if inactive_dimension == 0:
            inactive_energy = 0.0
        else:
            # Orthogonal projection without forming the dense d-by-d projector.
            inactive = scaled - active_coordinates @ self.U.T
            inactive_energy = (
                self.eta
                / inactive_dimension
                * np.sum(inactive * inactive, axis=1)
            )
        values = np.exp(-0.5 * (active_energy + inactive_energy))
        return _restore_output(values, scalar_input)

    def sample_normalized_on_domain(
        self,
        n_samples: int,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Sample from density proportional to this Gaussian on [-a, a]^d."""

        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        rng = _as_rng(random_state)
        samples = np.empty((n_samples, self.dimension), dtype=float)
        n_accepted = 0
        n_proposed = 0
        max_batch = max(1, 1_000_000 // self.dimension)
        max_proposals = max(100_000, 1000 * n_samples)

        # Uniform rejection is exact here because 0 < f(x) <= 1 on the box.
        while n_accepted < n_samples:
            remaining = n_samples - n_accepted
            batch_size = min(max_batch, max(1024, 4 * remaining))
            candidates = rng.uniform(
                -self.a,
                self.a,
                size=(batch_size, self.dimension),
            )
            n_proposed += batch_size
            accepted = candidates[
                rng.random(batch_size) < np.asarray(self(candidates), dtype=float)
            ]
            take = min(remaining, accepted.shape[0])
            if take:
                samples[n_accepted : n_accepted + take] = accepted[:take]
                n_accepted += take
            if n_accepted < n_samples and n_proposed >= max_proposals:
                raise RuntimeError(
                    "Gaussian rejection sampling acceptance is too low; "
                    "check active_mean and tau."
                )
        return samples


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
        (1.95, -1.0),
        (-1.2, 1.65),
        (1.4, 1.3),
    )
    angles = (20.0, -30.0, 60.0, 5.0)
    spectra = (
        (2.55, 0.125),
        (0.10, 5.00),
        (1.70, 1.7),
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


def default_high_dim_gaussian_mixture(
    dimension: int,
    beta: float = 0.0,
    random_state: RandomState = None,
) -> GaussianMixtureFunction:
    """Return the fixed two-component high-dimensional Gaussian benchmark."""

    dimension = _validate_dimension(dimension, minimum=1)
    rng = _as_rng(random_state)
    center = 0.5 * np.ones(dimension)
    components = tuple(
        GaussianFunction(
            dimension=dimension,
            beta=0.0,
            amplitude=1.0,
            sharpness=8.0,
            domain_scale=1.0,
            condition_number=GAUSSIAN_DEFAULT_CONDITION_NUMBER,
            q_mode="random",
            random_state=rng,
            mean=sign * center,
        )
        for sign in (-1.0, 1.0)
    )
    return GaussianMixtureFunction(
        components=components,
        weights=np.ones(2),
        beta=beta,
        normalize_weights=False,
    )


def sphere_radius_for_box_volume_fraction(
    bounds: np.ndarray,
    volume_fraction: float,
    center: np.ndarray | None = None,
    n_probe: int = 100_000,
    random_state: RandomState = None,
) -> float:
    """Estimate a sphere radius enclosing a fraction of a uniform box."""

    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}.")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite with lower values below upper values.")
    volume_fraction = float(volume_fraction)
    if not 0.0 < volume_fraction < 1.0:
        raise ValueError("volume_fraction must satisfy 0 < fraction < 1.")
    if n_probe < 2:
        raise ValueError("n_probe must be at least 2.")

    dimension = bounds.shape[0]
    center = (
        np.mean(bounds, axis=1)
        if center is None
        else _validate_vector(center, dimension, "center")
    )
    rng = _as_rng(random_state)
    distances = np.empty(int(n_probe), dtype=float)
    batch_size = max(1, min(int(n_probe), 1_000_000 // dimension))
    for start in range(0, int(n_probe), batch_size):
        stop = min(start + batch_size, int(n_probe))
        points = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(stop - start, dimension),
        )
        distances[start:stop] = np.linalg.norm(points - center, axis=1)
    return float(np.quantile(distances, volume_fraction))


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


@dataclass
class RastriginFunction:
    """Standard multimodal Rastrigin function."""

    dimension: int
    A: float = 10.0

    def __post_init__(self) -> None:
        self.dimension = _validate_dimension(self.dimension, minimum=1)
        self.A = _validate_finite_scalar(self.A, "A")
        if self.A < 0.0:
            raise ValueError("A must be nonnegative.")

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        X, scalar_input = _as_points(x, self.dimension)
        values = self.A * self.dimension + np.sum(
            X**2 - self.A * np.cos(2.0 * np.pi * X),
            axis=1,
        )
        return _restore_output(values, scalar_input)


def _paper_diagonal(dimension: int) -> np.ndarray:
    indices = np.arange(1, dimension + 1, dtype=float)
    return indices**-3


def _geometric_precision_spectrum(
    dimension: int,
    condition_number: float,
) -> np.ndarray:
    """Return a trace-one geometric spectrum with the requested condition."""

    if dimension == 1:
        return np.ones(1)
    indices = np.arange(dimension, dtype=float)
    spectrum = np.exp(
        -np.log(condition_number) * indices / (dimension - 1)
    )
    if np.any(spectrum == 0.0):
        raise ValueError("condition_number is too large to represent the spectrum.")
    return spectrum / np.sum(spectrum)


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
