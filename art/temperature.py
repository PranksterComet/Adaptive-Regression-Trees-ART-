"""Temperature scale estimates for soft oblique splitters."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Literal

import numpy as np


TemperatureMode = Literal["median_nn", "median_pairwise_scaled"]
NearestNeighborMethod = Literal["kdtree", "bruteforce"]
TemperatureStrategy = Literal["splitter", "fixed", "tune_root", "tune_node"]

DEFAULT_TEMPERATURE_GRID = (1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)


@dataclass(frozen=True)
class TemperatureConfig:
    """Node-level temperature scaling and optional tuning settings."""

    strategy: TemperatureStrategy = "fixed"
    scale_mode: TemperatureMode = "median_nn"
    c: float = 0.1
    c_values: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    validation_fraction: float = 0.2
    max_points: int | None = 512
    nn_method: NearestNeighborMethod | None = None
    bruteforce_dimension_threshold: int = 20

    def __post_init__(self) -> None:
        if self.strategy not in ("splitter", "fixed", "tune_root", "tune_node"):
            raise ValueError("Unknown temperature strategy.")
        if self.c <= 0.0:
            raise ValueError("Temperature c must be positive.")
        if not self.c_values or any(value <= 0.0 for value in self.c_values):
            raise ValueError("Temperature c_values must be nonempty and positive.")
        if not (0.0 < self.validation_fraction < 0.5):
            raise ValueError("validation_fraction must satisfy 0 < fraction < 0.5.")
        if self.max_points is not None and self.max_points < 2:
            raise ValueError("max_points must be at least 2 or None.")
        if self.bruteforce_dimension_threshold < 1:
            raise ValueError("bruteforce_dimension_threshold must be at least 1.")


def subsample_points(
    X: np.ndarray,
    max_points: int | None = 512,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    if max_points is None or X.shape[0] <= max_points:
        return X
    if max_points < 1:
        raise ValueError("max_points must be positive or None.")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=int(max_points), replace=False)
    return X[idx]


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=2)


def median_nearest_neighbor_distance(
    X: np.ndarray,
    method: NearestNeighborMethod = "kdtree",
) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("At least two points are required.")
    if method not in ("kdtree", "bruteforce"):
        raise ValueError("method must be 'kdtree' or 'bruteforce'.")

    if method == "kdtree":
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            warnings.warn(
                "scipy.spatial.cKDTree is unavailable; falling back to brute-force nearest-neighbor distances.",
                RuntimeWarning,
                stacklevel=2,
            )
            method = "bruteforce"
        else:
            distances, _ = cKDTree(X).query(X, k=2)
            return float(np.median(distances[:, 1]))

    distances = pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))


def median_pairwise_distance(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("At least two points are required.")

    distances = pairwise_distances(X)
    upper = distances[np.triu_indices(X.shape[0], k=1)]
    return float(np.median(upper))


def estimate_temperature(
    X: np.ndarray,
    mode: TemperatureMode,
    c: float = 1.0,
    max_points: int | None = 512,
    random_state: int | np.random.Generator | None = None,
    nn_method: NearestNeighborMethod = "kdtree",
) -> float:
    X_used = subsample_points(X, max_points=max_points, random_state=random_state)
    if c <= 0.0:
        raise ValueError("c must be positive.")
    if mode == "median_nn":
        base = median_nearest_neighbor_distance(X_used, method=nn_method)
    elif mode == "median_pairwise_scaled":
        base = median_pairwise_distance(X_used) * (X_used.shape[0] ** (-1.0 / X_used.shape[1]))
    else:
        raise ValueError("mode must be 'median_nn' or 'median_pairwise_scaled'.")

    temperature = float(c) * float(base)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("Estimated temperature is not positive and finite.")
    return temperature


def temperature_grid(
    X: np.ndarray,
    mode: TemperatureMode,
    c_values: list[float] | tuple[float, ...],
    max_points: int | None = 512,
    random_state: int | np.random.Generator | None = None,
    nn_method: NearestNeighborMethod = "kdtree",
) -> list[float]:
    temperatures = [
        estimate_temperature(
            X,
            mode=mode,
            c=float(c),
            max_points=max_points,
            random_state=random_state,
            nn_method=nn_method,
        )
        for c in c_values
    ]

    unique_temperatures = []
    for temperature in temperatures:
        if not any(np.isclose(temperature, existing) for existing in unique_temperatures):
            unique_temperatures.append(temperature)
    return unique_temperatures
