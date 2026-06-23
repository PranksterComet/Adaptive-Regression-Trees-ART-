"""Optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LineSearchResult:
    theta: np.ndarray
    value: float
    step_size: float
    success: bool
    n_backtracks: int


@dataclass
class AdaptiveAlpha:
    """Stateful controller for the next initial backtracking step size."""
    # If backtracking fails, smallest step*rho to try next


    alpha: float = 1.0
    alpha_min: float = 1e-12
    alpha_max: float = 1e3
    grow: float = 10.0 # grow if first backtrack succeeds
    recovery: float = 10.0 # multiplier to accepted step if many backtracks were needed
    heavy_backtrack_threshold: int = 8

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if self.alpha_min <= 0.0:
            raise ValueError("alpha_min must be positive.")
        if self.alpha_max < self.alpha_min:
            raise ValueError("alpha_max must be at least alpha_min.")
        if self.grow < 1.0:
            raise ValueError("grow must be at least 1.")
        if self.recovery < 1.0:
            raise ValueError("recovery must be at least 1.")
        if self.heavy_backtrack_threshold < 0:
            raise ValueError("heavy_backtrack_threshold must be nonnegative.")
        self.alpha = self._clip(self.alpha)

    def update(self, result: LineSearchResult, rho: float) -> float:
        """Update and return the next alpha0."""

        if not (0.0 < rho < 1.0):
            raise ValueError("rho must satisfy 0 < rho < 1.")

        if not result.success:
            self.alpha = self._clip(result.step_size * rho)
            return self.alpha

        if result.n_backtracks == 0:
            self.alpha = self._clip(result.step_size * self.grow)
        elif result.n_backtracks >= self.heavy_backtrack_threshold:
            self.alpha = self._clip(result.step_size * self.recovery)
        else:
            self.alpha = self._clip(result.step_size / rho)

        return self.alpha

    def _clip(self, alpha: float) -> float:
        return min(max(float(alpha), self.alpha_min), self.alpha_max)


def armijo_backtracking(
    value_fn: Callable[[np.ndarray], float],
    candidate_fn: Callable[[float], np.ndarray],
    current_value: float,
    directional_derivative: float,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_backtracks: int = 25,
) -> LineSearchResult:
    """Backtracking line search using the Armijo sufficient-decrease condition."""

    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive.")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must satisfy 0 < rho < 1.")
    if not (0.0 < c < 1.0):
        raise ValueError("c must satisfy 0 < c < 1.")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be nonnegative.")
    if directional_derivative >= 0.0:
        raise ValueError("directional_derivative must be negative for a descent direction.")

    alpha = float(alpha0)
    last_theta = candidate_fn(alpha)
    last_value = float(value_fn(last_theta))

    for n_backtracks in range(max_backtracks + 1):
        if n_backtracks > 0:
            alpha *= rho
            last_theta = candidate_fn(alpha)
            last_value = float(value_fn(last_theta))

        armijo_rhs = current_value + c * alpha * directional_derivative
        if last_value <= armijo_rhs:
            return LineSearchResult(
                theta=np.asarray(last_theta, dtype=float),
                value=last_value,
                step_size=alpha,
                success=True,
                n_backtracks=n_backtracks,
            )

    return LineSearchResult(
        theta=np.asarray(last_theta, dtype=float),
        value=last_value,
        step_size=alpha,
        success=False,
        n_backtracks=max_backtracks,
    )
