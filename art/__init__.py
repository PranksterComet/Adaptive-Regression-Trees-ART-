"""Adaptive regression tree package."""

from .domain import BoxDomain, PolytopeRegion, split_region
from .metrics import (
    max_pointwise_relative_error,
    mean_squared_error,
    median_pointwise_relative_error,
    pointwise_relative_error,
    relative_l2_error,
    sum_squared_error,
)
from .models import (
    AffineRidgeModel,
    KernelRidgeModel,
    PolynomialRidgeModel,
    RegressionModel,
    WeightedRegressionModel,
    scaled_ridge_from_gram,
    solve_weighted_ridge,
)
from .objectives import (
    SoftObjectiveResult,
    SoftObliqueRidgeObjective,
    sigmoid,
)
from .optimizers import AdaptiveAlpha, LineSearchResult, armijo_backtracking
from .sampling import (
    HitAndRunSampler,
    Sampler,
    UniformBoxSampler,
    find_feasible_point,
    infer_box_bounds,
    sample_uniform_box,
)
from .splitters import (
    HingeAffineSplitter,
    SoftObliqueSplitter,
    SplitResult,
    project_unit_w,
    project_unit_w_gradient,
)
from .temperature import (
    estimate_temperature,
    median_nearest_neighbor_distance,
    median_pairwise_distance,
    pairwise_distances,
    subsample_points,
    temperature_grid,
)
from .tree import LeafNode, RegressionTree, SplitNode

__all__ = [
    "AffineRidgeModel",
    "AdaptiveAlpha",
    "BoxDomain",
    "HingeAffineSplitter",
    "HitAndRunSampler",
    "KernelRidgeModel",
    "LeafNode",
    "LineSearchResult",
    "PolytopeRegion",
    "PolynomialRidgeModel",
    "RegressionModel",
    "RegressionTree",
    "Sampler",
    "SoftObjectiveResult",
    "SoftObliqueSplitter",
    "SoftObliqueRidgeObjective",
    "SplitNode",
    "SplitResult",
    "UniformBoxSampler",
    "WeightedRegressionModel",
    "armijo_backtracking",
    "estimate_temperature",
    "find_feasible_point",
    "infer_box_bounds",
    "max_pointwise_relative_error",
    "mean_squared_error",
    "median_nearest_neighbor_distance",
    "median_pairwise_distance",
    "median_pointwise_relative_error",
    "pairwise_distances",
    "pointwise_relative_error",
    "project_unit_w",
    "project_unit_w_gradient",
    "relative_l2_error",
    "sample_uniform_box",
    "scaled_ridge_from_gram",
    "sigmoid",
    "solve_weighted_ridge",
    "split_region",
    "subsample_points",
    "sum_squared_error",
    "temperature_grid",
]
