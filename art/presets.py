"""Recommended defaults for adaptive regression tree experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .models import (
    DEFAULT_AUTO_RCOND_THRESHOLD,
    RegressionModel,
    RidgeSolver,
    model_effective_dimension,
)
from .temperature import (
    DEFAULT_TEMPERATURE_GRID,
    TemperatureMode,
    TemperatureStrategy,
)


NearestNeighborSelection = Literal["auto", "kdtree", "bruteforce"]
TreeSplitterName = Literal["soft_oblique", "hrt"]


@dataclass(frozen=True)
class SoftObliquePreset:
    """Default projected-gradient and split-validity settings."""

    temperature_placeholder: float = 0.1 #placeholder to initialize splitter, will be overwritten
    max_iters: int = 200
    grad_atol: float = 1e-8
    grad_rtol: float = 1e-5
    min_side_points: int | None = None
    min_side_fraction: float = 0.0
    n_restarts: int = 1
    alpha0: float = 1.0
    rho: float = 0.5
    armijo_c: float = 1e-4
    max_backtracks: int = 25
    adaptive_alpha: bool = True
    alpha_min: float = 1e-12
    alpha_max: float = 1e8
    alpha_grow: float = 10.0
    alpha_recovery: float = 10.0
    heavy_backtrack_threshold: int = 8
    max_line_search_failures: int = 5
    weight_floor: float = 1e-12
    refit_during_line_search: bool = False


@dataclass(frozen=True)
class HingeAffinePreset:
    """Default settings specific to the affine hinge splitter."""

    mode: Literal["max", "min", "both"] = "both"
    mu: float = 1.0
    tol: float = 1e-6
    init_scale: float = 1e-2


@dataclass(frozen=True)
class SamplingPreset:
    """Default hit-and-run settings for tree nodes."""

    burn_in: int = 0
    thinning: int = 20
    feasibility_tol: float = 1e-10
    max_feasible_tries: int = 20_000
    isotropic_sampling: bool = True
    isotropic_pilot_multiplier: int = 50
    direction_eigenvalue_floor: float = 1e-2


@dataclass(frozen=True)
class TemperaturePreset:
    """Default node-local temperature scaling settings."""

    strategy: TemperatureStrategy = "fixed"
    scale_mode: TemperatureMode = "median_nn"
    c: float = 1.0
    c_values: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    validation_fraction: float = 0.2
    max_points: int | None = 512
    nn_method: NearestNeighborSelection = "auto"
    bruteforce_dimension_threshold: int = 20


@dataclass(frozen=True)
class TreePreset:
    """Shared defaults for the 2D and high-dimensional tree harnesses."""

    leaf_degree: int = 1
    leaf_include_bias: bool = True
    splitter_name: TreeSplitterName = "soft_oblique"
    error_metric: str = "max_pointwise_relative"
    error_tolerance: float = 1e-2
    relative_error_floor: float = 1e-12
    max_depth: int = 10
    sample_multiplier: int = 50
    sample_count: int | None = None
    ridge: float = 0.0
    ridge_solver: RidgeSolver = "auto"
    auto_rcond_threshold: float = DEFAULT_AUTO_RCOND_THRESHOLD
    min_split_gain: float = 0.0
    min_relative_split_gain: float = 1e-3
    max_retries_on_failure: int = 3
    exact_box_root: bool = True
    store_samples: bool = False
    store_diagnostics: bool = True
    profile_build_timing: bool = False
    oracle_vectorized: bool = True
    sampling: SamplingPreset = field(default_factory=SamplingPreset)
    temperature: TemperaturePreset = field(default_factory=TemperaturePreset)
    splitter: SoftObliquePreset = field(default_factory=SoftObliquePreset)
    hinge_splitter: HingeAffinePreset = field(default_factory=HingeAffinePreset)


DEFAULT_TREE_PRESET = TreePreset()


def resolve_min_side_points(
    configured_value: int | None,
    model: RegressionModel,
    input_dimension: int,
) -> tuple[int, str]:
    """Resolve an explicit minimum or use the model's effective dimension."""

    if configured_value is None:
        return model_effective_dimension(model, input_dimension), "effective_dimension"
    value = int(configured_value)
    if value < 1:
        raise ValueError("min_side_points must be at least 1.")
    return value, "explicit"
