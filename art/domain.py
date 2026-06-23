"""Domain and polytope region utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PolytopeRegion:
    """A convex polytope represented by linear inequalities A x <= b."""

    A: np.ndarray
    b: np.ndarray
    depth: int = 0
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float).reshape(-1)

        if A.ndim != 2:
            raise ValueError(f"A must have shape (m, d), got {A.shape}.")
        if b.ndim != 1:
            raise ValueError(f"b must have shape (m,), got {b.shape}.")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"A and b must have compatible rows, got {A.shape[0]} and {b.shape[0]}."
            )

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @property
    def dimension(self) -> int:
        return int(self.A.shape[1])

    def contains(self, X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Return a boolean mask indicating whether points satisfy A x <= b."""

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(f"X must have shape (n, {self.dimension}), got {X.shape}.")
        return np.all(X @ self.A.T <= self.b[None, :] + tol, axis=1)


@dataclass(frozen=True)
class BoxDomain:
    """Axis-aligned box domain with bounds shaped (d, 2)."""

    bounds: np.ndarray

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}.")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each bound must satisfy low < high.")
        object.__setattr__(self, "bounds", bounds)

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])

    def as_region(self, depth: int = 0, tag: Optional[str] = "root") -> PolytopeRegion:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        A = np.vstack([np.eye(self.dimension), -np.eye(self.dimension)])
        b = np.concatenate([highs, -lows])
        return PolytopeRegion(A=A, b=b, depth=depth, tag=tag)


def split_region(
    region: PolytopeRegion,
    w: np.ndarray,
    z: float,
    depth: Optional[int] = None,
    left_tag: Optional[str] = None,
    right_tag: Optional[str] = None,
) -> tuple[PolytopeRegion, PolytopeRegion]:
    """Split a region by w^T x - z, returning left (< 0) and right (>= 0)."""

    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != region.dimension:
        raise ValueError(f"w must have shape ({region.dimension},), got {w.shape}.")

    child_depth = region.depth + 1 if depth is None else int(depth)

    left_A = np.vstack([region.A, w.reshape(1, -1)])
    left_b = np.concatenate([region.b, np.array([float(z)])])

    right_A = np.vstack([region.A, -w.reshape(1, -1)])
    right_b = np.concatenate([region.b, np.array([-float(z)])])

    left = PolytopeRegion(A=left_A, b=left_b, depth=child_depth, tag=left_tag)
    right = PolytopeRegion(A=right_A, b=right_b, depth=child_depth, tag=right_tag)
    return left, right
