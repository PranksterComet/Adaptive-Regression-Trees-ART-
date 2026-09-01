"""Sampling routines for domains and polytope regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from .domain import BoxDomain, PolytopeRegion


RandomState = int | np.random.Generator | None
ThinningCandidateMode = Literal["linear", "powers_of_two"]
ProbeMode = Literal["coordinates", "random", "both"]


def as_rng(random_state: RandomState = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


@runtime_checkable
class Sampler(Protocol):
    """Protocol for region samplers."""

    def sample(
        self,
        region: PolytopeRegion,
        n: int,
        random_state: RandomState = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


def sample_uniform_box(
    bounds: np.ndarray,
    n: int,
    random_state: RandomState = None,
) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}.")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each bound must satisfy low < high.")
    if n < 0:
        raise ValueError("n must be nonnegative.")

    rng = as_rng(random_state)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return rng.uniform(lows, highs, size=(int(n), bounds.shape[0]))


def sample_covariance_eigendecomposition(
    samples: np.ndarray,
    *,
    assume_centered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvectors and eigenvalues of the sample covariance matrix."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"samples must have shape (n_samples, d), got {values.shape}.")
    if values.shape[0] < 2:
        raise ValueError("samples must contain at least two points.")
    if values.shape[1] < 1:
        raise ValueError("samples must contain at least one coordinate.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")

    centered = values if assume_centered else values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors, eigenvalues


def floor_covariance_eigenvalues(
    eigenvalues: np.ndarray,
    floor_ratio: float,
) -> tuple[np.ndarray, float]:
    """Apply a trace-scaled positive floor to covariance eigenvalues."""

    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("eigenvalues must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite.")
    if not np.isfinite(floor_ratio) or floor_ratio <= 0.0:
        raise ValueError("floor_ratio must be positive.")

    mean_eigenvalue = float(np.mean(values))
    if mean_eigenvalue <= 0.0:
        raise ValueError("eigenvalues must have a positive mean before flooring.")
    eigenvalue_floor = float(floor_ratio * mean_eigenvalue)
    return np.maximum(values, eigenvalue_floor), eigenvalue_floor


@dataclass
class UniformBoxSampler:
    """Exact uniform sampler for axis-aligned boxes."""

    def sample(
        self,
        domain: BoxDomain,
        n: int,
        random_state: RandomState = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        del x0
        return sample_uniform_box(domain.bounds, n, random_state=random_state)


def infer_box_bounds(region: PolytopeRegion, tol: float = 1e-12) -> np.ndarray | None:
    """Infer box bounds from +/- coordinate constraints when present."""

    lows = np.full(region.dimension, -np.inf)
    highs = np.full(region.dimension, np.inf)

    for row, bound in zip(region.A, region.b):
        nz = np.flatnonzero(np.abs(row) > tol)
        if nz.size != 1:
            continue

        idx = int(nz[0])
        coef = row[idx]
        if coef > 0:
            highs[idx] = min(highs[idx], bound / coef)
        else:
            lows[idx] = max(lows[idx], bound / coef)

    if np.any(~np.isfinite(lows)) or np.any(~np.isfinite(highs)):
        return None
    if np.any(lows >= highs):
        return None
    return np.column_stack([lows, highs])


def find_feasible_point(
    region: PolytopeRegion,
    bounds: np.ndarray | None = None,
    max_tries: int = 20000,
    random_state: RandomState = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Find a feasible point by rejection sampling from a bounding box."""

    if bounds is None:
        bounds = infer_box_bounds(region)
    if bounds is None:
        raise ValueError("bounds are required when they cannot be inferred from the region.")

    rng = as_rng(random_state)
    for _ in range(max_tries):
        x = sample_uniform_box(bounds, 1, random_state=rng)[0]
        if bool(region.contains(x, tol=tol)[0]):
            return x

    raise RuntimeError("Could not find a feasible point inside the region.")


def make_thinning_candidates(
    max_thinning: int,
    mode: ThinningCandidateMode = "linear",
) -> np.ndarray:
    """Return positive thinning candidates up to max_thinning."""

    max_thinning = int(max_thinning)
    if max_thinning < 1:
        raise ValueError("max_thinning must be at least 1.")
    if mode == "linear":
        return np.arange(1, max_thinning + 1, dtype=int)
    if mode == "powers_of_two":
        candidates = []
        value = 1
        while value <= max_thinning:
            candidates.append(value)
            value *= 2
        return np.asarray(candidates, dtype=int)
    raise ValueError("mode must be 'linear' or 'powers_of_two'.")


def autocorrelation_by_lag(series: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Estimate signed autocorrelation for each lag and scalar series.

    Uses one global mean and estimates rho(k) = gamma(k) / gamma(0), where
    gamma(k) averages over the overlapping lag-k pairs.
    """

    values = np.asarray(series, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"series must have shape (n_steps,) or (n_steps, n_series), got {values.shape}.")
    if values.shape[0] < 2:
        raise ValueError("series must contain at least two states.")
    if not np.all(np.isfinite(values)):
        raise ValueError("series must contain only finite values.")

    lags_array = np.asarray(lags, dtype=int).reshape(-1)
    if lags_array.size == 0:
        raise ValueError("lags must contain at least one value.")
    if np.any(lags_array < 1):
        raise ValueError("lags must be atleast 1.")
    if np.any(lags_array >= values.shape[0]):
        raise ValueError("all lags must be smaller than the series length.")

    centered = values - np.mean(values, axis=0, keepdims=True)
    scales = np.max(np.abs(centered), axis=0)
    nonconstant = scales > 0.0
    normalized = np.zeros_like(centered)
    np.divide(centered, scales, out=normalized, where=nonconstant)
    variance = np.mean(normalized * normalized, axis=0)
    autocorr = np.zeros((lags_array.size, values.shape[1]), dtype=float)

    for i, lag in enumerate(lags_array):
        lag_covariance = np.mean(normalized[:-lag] * normalized[lag:], axis=0)
        autocorr[i] = np.divide(
            lag_covariance,
            variance,
            out=np.zeros_like(lag_covariance, dtype=float),
            where=nonconstant,
        )

    return autocorr


def estimate_thinning_from_chain(
    chain: np.ndarray,
    candidate_thinnings: Sequence[int] | None = None,
    max_thinning: int | None = None,
    candidate_mode: ThinningCandidateMode = "linear",
    acf_threshold: float = 0.1,
    stable_window: int = 3,
    probe_mode: ProbeMode = "both",
    num_probes: int = 16,
    whiten: bool = False,
    covariance_floor: float = 1e-12,
    random_state: RandomState = None,
) -> tuple[int, dict[str, object]]:
    """Estimate a thinning lag from autocorrelation decay of a pilot chain.

    The chain is treated as consecutive hit-and-run states. Autocorrelation is
    measured on linear probes of the geometry rather than on oracle values.
    For probe_mode="both", num_probes is the desired total number of probes;
    all coordinate probes are included, with random probes added if needed.
    """

    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 2:
        raise ValueError(f"chain must have shape (n_steps, d), got {chain.shape}.")
    if chain.shape[0] < 3:
        raise ValueError("chain must contain at least three states.")
    if not (0.0 <= acf_threshold < 1.0):
        raise ValueError("acf_threshold must satisfy 0 <= threshold < 1.")
    if stable_window < 1:
        raise ValueError("stable_window must be at least 1.")
    if num_probes < 0:
        raise ValueError("num_probes must be nonnegative.")
    if covariance_floor <= 0.0:
        raise ValueError("covariance_floor must be positive.")

    max_valid_lag = max(1, chain.shape[0] // 2)
    if max_thinning is None:
        max_lag = max_valid_lag
    else:
        max_lag = int(max_thinning)
        if max_lag < 1:
            raise ValueError("max_thinning must be at least 1 when provided.")
    max_lag = min(max_lag, max_valid_lag)

    candidates = _resolve_thinning_candidates(
        candidate_thinnings=candidate_thinnings,
        max_lag=max_lag,
        candidate_mode=candidate_mode,
    )
    rng = as_rng(random_state)
    transformed, transform_metadata = _transform_chain_for_acf(
        chain,
        whiten=whiten,
        covariance_floor=covariance_floor,
    )
    probes = _linear_probes(
        dimension=chain.shape[1],
        mode=probe_mode,
        num_probes=num_probes,
        rng=rng,
    )
    num_coordinate_probes = chain.shape[1] if probe_mode in ("coordinates", "both") else 0
    num_random_probes = int(probes.shape[0] - num_coordinate_probes)
    probe_series = transformed @ probes.T
    autocorr_by_lag = autocorrelation_by_lag(probe_series, candidates)
    rho_by_lag = np.max(np.abs(autocorr_by_lag), axis=1)

    selected_index = len(candidates) - 1
    window_size = min(int(stable_window), len(candidates))
    for idx in range(len(candidates) - window_size + 1):
        if float(np.max(rho_by_lag[idx : idx + window_size])) <= acf_threshold:
            selected_index = idx
            break

    metadata: dict[str, object] = {
        "candidate_thinnings": candidates.tolist(),
        "autocorrelation_by_lag": autocorr_by_lag.tolist(),
        "max_abs_autocorrelation": rho_by_lag.tolist(),
        "selected_index": selected_index,
        "acf_threshold": acf_threshold,
        "stable_window": stable_window,
        "effective_stable_window": window_size,
        "probe_mode": probe_mode,
        "num_coordinate_probes": num_coordinate_probes,
        "num_random_probes": num_random_probes,
        "num_total_probes": int(probes.shape[0]),
        "whiten": whiten,
        "chain_length": int(chain.shape[0]),
        "max_lag": int(max_lag),
        **transform_metadata,
    }
    return int(candidates[selected_index]), metadata


def _resolve_thinning_candidates(
    candidate_thinnings: Sequence[int] | None,
    max_lag: int,
    candidate_mode: ThinningCandidateMode,
) -> np.ndarray:
    if candidate_thinnings is None:
        candidates = make_thinning_candidates(max_lag, mode=candidate_mode)
    else:
        candidates = np.asarray(candidate_thinnings, dtype=int).reshape(-1)
        if candidates.size == 0:
            raise ValueError("candidate_thinnings must contain at least one value.")
        if np.any(candidates < 1):
            raise ValueError("candidate_thinnings must be positive.")
        candidates = np.unique(candidates)
        candidates = candidates[candidates <= max_lag]

    if candidates.size == 0:
        raise ValueError("No thinning candidates are valid for this chain length.")
    return candidates.astype(int, copy=False)


def _transform_chain_for_acf(
    chain: np.ndarray,
    whiten: bool,
    covariance_floor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not whiten:
        return chain, {"whitening_eig_floor": None, "whitening_condition_number": None}

    centered = chain - np.mean(chain, axis=0, keepdims=True)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        return centered, {"whitening_eig_floor": 0.0, "whitening_condition_number": None}

    scaled = centered / scale
    eigvecs, eigvals = sample_covariance_eigendecomposition(
        scaled,
        assume_centered=True,
    )
    eig_max = float(np.max(eigvals)) if eigvals.size else 0.0
    relative_floor = max(covariance_floor, np.finfo(float).eps * chain.shape[1])
    eig_floor = float(relative_floor * eig_max)
    eigvals_safe = np.maximum(eigvals, eig_floor)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.T
    condition_number = float(np.max(eigvals_safe) / np.min(eigvals_safe))
    return scaled @ inv_sqrt.T, {
        "whitening_eig_floor": eig_floor,
        "whitening_condition_number": condition_number,
    }


def _linear_probes(
    dimension: int,
    mode: ProbeMode,
    num_probes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode not in ("coordinates", "random", "both"):
        raise ValueError("probe_mode must be 'coordinates', 'random', or 'both'.")

    probes = []
    if mode in ("coordinates", "both"):
        probes.append(np.eye(dimension))
    n_random = int(num_probes)
    if mode == "both":
        n_random = max(0, int(num_probes) - dimension)

    if mode in ("random", "both") and n_random > 0:
        random_probes = rng.normal(size=(n_random, dimension))
        norms = np.linalg.norm(random_probes, axis=1)
        keep = norms > 1e-14
        if np.any(keep):
            probes.append(random_probes[keep] / norms[keep, None])

    if not probes:
        raise ValueError("At least one probe is required.")
    return np.vstack(probes)


@dataclass
class HitAndRunSampler:
    """Hit-and-run sampler for convex polytopes A x <= b."""

    burn_in: int = 500
    thinning: int = 10
    feasibility_tol: float = 1e-10
    max_feasible_tries: int = 20000
    bounds: np.ndarray | None = None
    direction_eigenvectors: np.ndarray | None = None
    direction_eigenvalues: np.ndarray | None = None
    direction_eigenvalue_floor: float = 1e-4  # At most 1e2 directional aspect ratio.

    def sample(
        self,
        region: PolytopeRegion,
        n: int,
        random_state: RandomState = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative.")
        if self.thinning < 1:
            raise ValueError("thinning must be at least 1.")

        rng = as_rng(random_state)
        eigenvectors, sqrt_eigenvalues = self._direction_spectrum(region.dimension)
        if x0 is None:
            x = find_feasible_point(
                region,
                bounds=self.bounds,
                max_tries=self.max_feasible_tries,
                random_state=rng,
                tol=self.feasibility_tol,
            )
        else:
            x = np.asarray(x0, dtype=float).reshape(-1)
            if x.shape[0] != region.dimension:
                raise ValueError(f"x0 must have shape ({region.dimension},), got {x.shape}.")
            if not bool(region.contains(x, tol=self.feasibility_tol)[0]):
                raise ValueError("x0 is not feasible for the region.")

        samples = []
        total_steps = self.burn_in + int(n) * self.thinning

        for step in range(total_steps):
            direction = rng.normal(size=region.dimension)
            if eigenvectors is not None:
                direction = eigenvectors @ (sqrt_eigenvalues * direction)
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0.0:
                continue
            direction = direction / direction_norm

            t_low, t_high = self._step_interval(region, x, direction)
            step_size = rng.uniform(t_low, t_high)
            x = x + step_size * direction

            if step >= self.burn_in and (step - self.burn_in) % self.thinning == 0:
                samples.append(x.copy())

        return np.asarray(samples, dtype=float)

    def _direction_spectrum(
        self,
        dimension: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        eigenvectors = self.direction_eigenvectors
        eigenvalues = self.direction_eigenvalues
        if eigenvectors is None and eigenvalues is None:
            return None, None
        if eigenvectors is None or eigenvalues is None:
            raise ValueError(
                "direction_eigenvectors and direction_eigenvalues must be provided together."
            )

        eigenvectors = np.asarray(eigenvectors, dtype=float)
        eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
        if eigenvectors.shape != (dimension, dimension):
            raise ValueError(
                "direction_eigenvectors must have shape "
                f"{(dimension, dimension)}, got {eigenvectors.shape}."
            )
        if eigenvalues.shape != (dimension,):
            raise ValueError(
                f"direction_eigenvalues must have shape ({dimension},), "
                f"got {eigenvalues.shape}."
            )
        if not np.all(np.isfinite(eigenvectors)) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("Direction eigenvectors and eigenvalues must be finite.")
        if not np.allclose(
            eigenvectors.T @ eigenvectors,
            np.eye(dimension),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("direction_eigenvectors must be orthonormal.")

        safe_eigenvalues, _ = floor_covariance_eigenvalues(
            eigenvalues,
            self.direction_eigenvalue_floor,
        )
        sqrt_eigenvalues = np.sqrt(safe_eigenvalues)
        sqrt_eigenvalues /= np.max(sqrt_eigenvalues)
        return eigenvectors, sqrt_eigenvalues

    def _step_interval(
        self,
        region: PolytopeRegion,
        x: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[float, float]:
        Ax = region.A @ x
        Ad = region.A @ direction
        t_low = -np.inf
        t_high = np.inf

        for a_dot_d, a_dot_x, bound in zip(Ad, Ax, region.b):
            if abs(a_dot_d) <= 1e-14:
                if a_dot_x > bound + self.feasibility_tol:
                    raise RuntimeError("Current point is infeasible.")
                continue

            t = (bound - a_dot_x) / a_dot_d
            if a_dot_d > 0:
                t_high = min(t_high, t)
            else:
                t_low = max(t_low, t)

        if t_low > t_high:
            raise RuntimeError("Empty hit-and-run step interval.")
        return float(t_low), float(t_high)
