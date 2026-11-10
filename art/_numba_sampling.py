"""Private compiled kernels for hit-and-run sampling."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _advance_precomputed_chains(
    points: np.ndarray,
    slack: np.ndarray,
    directions: np.ndarray,
    constraint_products: np.ndarray,
    uniforms: np.ndarray,
    step: int,
) -> None:
    n_chains, dimension = points.shape
    n_constraints = slack.shape[1]
    for chain in range(n_chains):
        t_low = -np.inf
        t_high = np.inf
        for constraint in range(n_constraints):
            product = constraint_products[step, chain, constraint]
            if product < 0.0:
                ratio = slack[chain, constraint] / product
                if ratio > t_low:
                    t_low = ratio
            elif product > 0.0:
                ratio = slack[chain, constraint] / product
                if ratio < t_high:
                    t_high = ratio

        step_size = t_low + uniforms[step, chain] * (t_high - t_low)
        for coordinate in range(dimension):
            points[chain, coordinate] += (
                step_size * directions[step, chain, coordinate]
            )
        for constraint in range(n_constraints):
            slack[chain, constraint] -= (
                step_size * constraint_products[step, chain, constraint]
            )


@njit(cache=True)
def sample_precomputed_chains(
    points: np.ndarray,
    slack: np.ndarray,
    directions: np.ndarray,
    constraint_products: np.ndarray,
    uniforms: np.ndarray,
    burn_in: int,
    thinning: int,
    samples: np.ndarray,
) -> None:
    """Run complete transition rounds and write chain states to ``samples``."""

    step = 0
    for _ in range(burn_in):
        _advance_precomputed_chains(
            points,
            slack,
            directions,
            constraint_products,
            uniforms,
            step,
        )
        step += 1

    n_chains = points.shape[0]
    n_rounds = samples.shape[0] // n_chains
    output = 0
    for _ in range(n_rounds):
        for _ in range(thinning):
            _advance_precomputed_chains(
                points,
                slack,
                directions,
                constraint_products,
                uniforms,
                step,
            )
            step += 1
        for chain in range(n_chains):
            for coordinate in range(points.shape[1]):
                samples[output + chain, coordinate] = points[chain, coordinate]
        output += n_chains


@njit(cache=True, inline="always")
def _advance_chain_major(
    points: np.ndarray,
    slack: np.ndarray,
    directions: np.ndarray,
    constraint_products: np.ndarray,
    uniforms: np.ndarray,
    chain: int,
    step: int,
) -> None:
    t_low = -np.inf
    t_high = np.inf
    for constraint in range(slack.shape[1]):
        product = constraint_products[chain, step, constraint]
        if product < 0.0:
            ratio = slack[chain, constraint] / product
            if ratio > t_low:
                t_low = ratio
        elif product > 0.0:
            ratio = slack[chain, constraint] / product
            if ratio < t_high:
                t_high = ratio

    step_size = t_low + uniforms[chain, step] * (t_high - t_low)
    for coordinate in range(points.shape[1]):
        points[chain, coordinate] += (
            step_size * directions[chain, step, coordinate]
        )
    for constraint in range(slack.shape[1]):
        slack[chain, constraint] -= (
            step_size * constraint_products[chain, step, constraint]
        )


@njit(cache=True, parallel=True)
def sample_precomputed_chains_parallel(
    points: np.ndarray,
    slack: np.ndarray,
    directions: np.ndarray,
    constraint_products: np.ndarray,
    uniforms: np.ndarray,
    burn_in: int,
    thinning: int,
    samples: np.ndarray,
) -> None:
    """Advance independent chain-major trajectories across CPU threads."""

    n_chains = points.shape[0]
    n_rounds = samples.shape[0] // n_chains
    for chain in prange(n_chains):
        step = 0
        for _ in range(burn_in):
            _advance_chain_major(
                points,
                slack,
                directions,
                constraint_products,
                uniforms,
                chain,
                step,
            )
            step += 1

        for output_round in range(n_rounds):
            for _ in range(thinning):
                _advance_chain_major(
                    points,
                    slack,
                    directions,
                    constraint_products,
                    uniforms,
                    chain,
                    step,
                )
                step += 1
            output = output_round * n_chains + chain
            for coordinate in range(points.shape[1]):
                samples[output, coordinate] = points[chain, coordinate]
