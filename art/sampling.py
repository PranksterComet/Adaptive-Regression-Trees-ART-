"""Sampling routines for domains and polytope regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .domain import BoxDomain, PolytopeRegion


RandomState = int | np.random.Generator | None


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


@dataclass
class HitAndRunSampler:
    """Hit-and-run sampler for convex polytopes A x <= b."""

    burn_in: int = 500
    thinning: int = 10
    feasibility_tol: float = 1e-10
    max_feasible_tries: int = 20000
    bounds: np.ndarray | None = None

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
