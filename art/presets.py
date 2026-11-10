"""Recommended defaults for adaptive regression tree experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from .models import (
    DEFAULT_AUTO_RCOND_THRESHOLD,
    RidgeSolver,
)
from .splitters import Optimizer
from .temperature import (
    DEFAULT_TEMPERATURE_GRID,
    TemperatureMode,
    TemperatureStrategy,
)


NearestNeighborSelection = Literal["auto", "kdtree", "bruteforce"]
TreeSplitterName = Literal["soft_oblique", "hrt"]
TreeSamplerName = Literal["hit_and_run", "parallel_numba"]


@dataclass(frozen=True)
class SoftObliquePreset:
    """Default projected-gradient and split-validity settings."""

    temperature_placeholder: float = 0.1 #placeholder to initialize splitter, will be overwritten
    max_iters: int = 200
    grad_atol: float = 1e-8
    grad_rtol: float = 1e-5
    min_side_points: int = 1
    min_side_fraction: float = 0.05
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
    optimizer: Optimizer = "cpu"
    backtrack_batch_size: int = 8
    Xw_refresh: int = 32
    qr_parallel: bool = True


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

    sampler_name: TreeSamplerName = "hit_and_run"
    burn_in: int = 0
    thinning: int = 20
    n_chains: int = 50
    threads: int = 10
    parallel_random: bool = True
    feasibility_tol: float = 1e-10
    max_feasible_tries: int = 20_000
    isotropic_sampling: bool = False
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
    traversal: Literal["dfs", "bfs"] = "dfs"
    node_batch_size: int = 20
    batch_reorder: bool = True
    error_metric: str = "max_pointwise_relative"
    error_tolerance: float = 1e-2
    relative_error_floor: float = 1e-12
    max_depth: int = 10
    sample_multiplier: int = 50
    sample_count: int | None = None
    ridge: float = 0.0
    ridge_solver: RidgeSolver = "auto"
    auto_rcond_threshold: float = DEFAULT_AUTO_RCOND_THRESHOLD
    check_split_gain: bool = False
    min_split_gain: float = 0.0
    min_relative_split_gain: float = 0.0
    max_retries_on_failure: int = 3
    min_side_fallback: bool = True
    validation_fraction: float = 0.0
    exact_box_root: bool = True
    store_samples: bool = False
    store_leaf_representative: bool = False
    store_diagnostics: bool = True
    profile_build_timing: bool = False
    oracle_vectorized: bool = True
    sampling: SamplingPreset = field(default_factory=SamplingPreset)
    temperature: TemperaturePreset = field(default_factory=TemperaturePreset)
    splitter: SoftObliquePreset = field(default_factory=SoftObliquePreset)
    hinge_splitter: HingeAffinePreset = field(default_factory=HingeAffinePreset)


DEFAULT_TREE_PRESET = TreePreset()
BFS_GPU_TREE_PRESET = replace(
    DEFAULT_TREE_PRESET, traversal="bfs",
    splitter=replace(DEFAULT_TREE_PRESET.splitter, optimizer="cupy_batched_cholesky"),
)
