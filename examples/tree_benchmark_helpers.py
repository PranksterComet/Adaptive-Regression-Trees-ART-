"""Shared construction, evaluation, and reporting for tree benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np

from art.builder import RegressionTreeBuilder, TreeBuildResult
from art.domain import BoxDomain
from art.metrics import (
    max_pointwise_relative_error,
    mean_squared_error,
    median_pointwise_relative_error,
    pointwise_relative_error,
    pointwise_relative_error_quantile,
    relative_l2_error,
)
from art.presets import DEFAULT_TREE_PRESET
from art.sampling import (
    HitAndRunSampler,
    ParallelNumbaBatchedHitAndRunSampler,
    sample_uniform_box,
)
from art.splitters import OPTIMIZERS, HingeAffineSplitter, SoftObliqueSplitter, Splitter
from art.temperature import TemperatureConfig
from art.tree import LeafNode, RegressionTree, SplitNode, TreeNode

from examples.test_helpers import make_model_template, parse_csv_floats


TREE_DEFAULTS = DEFAULT_TREE_PRESET
SAMPLING_DEFAULTS = TREE_DEFAULTS.sampling
TEMPERATURE_DEFAULTS = TREE_DEFAULTS.temperature
SPLITTER_DEFAULTS = TREE_DEFAULTS.splitter
HRT_DEFAULTS = TREE_DEFAULTS.hinge_splitter

ERROR_METRICS = {
    "relative_l2": relative_l2_error,
    "mse": mean_squared_error,
    "median_pointwise_relative": median_pointwise_relative_error,
    "max_pointwise_relative": max_pointwise_relative_error,
}
POINTWISE_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
EXPECTED_LEAF_STATUSES = (
    "tolerance_met",
    "max_depth",
    "invalid_split",
    "min_side_points",
    "insufficient_split_gain",
    "insufficient_relative_split_gain",
    "nonfinite_split",
    "optimizer_failed_without_progress",
    "temperature_tuning_failed",
    "sampling_failed",
)


@dataclass(frozen=True)
class TreeBenchmarkRun:
    builder: RegressionTreeBuilder
    build_result: TreeBuildResult
    tree: RegressionTree
    model: object
    leaf_include_bias: bool
    min_side_points: int
    build_seconds: float


@dataclass(frozen=True)
class TreeTestEvaluation:
    metrics: dict[str, float]
    pointwise_relative_quantiles: dict[float, float]
    selected_point_errors: np.ndarray
    selected_point_error_label: str
    oracle_min: float
    oracle_max: float
    timings: dict[str, float]
    bad_node_test_fractions: dict[str, float] | None = None


def build_timing_report_lines(
    timing: dict[str, float] | None,
    *,
    include_model_refits: bool = True,
) -> tuple[str, ...]:
    """Format optional aggregate build profiling with total-time percentages."""

    if timing is None:
        return ("enabled: False",)
    lines = [
        "enabled: True",
        f"total_build: {timing['total_build_seconds']:.6f} (100.00%)",
        (
            f"sampling: {timing['sampling_seconds']:.6f} "
            f"({timing['sampling_percent']:.2f}%)"
        ),
        (
            f"splitter: {timing['splitter_seconds']:.6f} "
            f"({timing['splitter_percent']:.2f}%)"
        ),
    ]
    if include_model_refits:
        lines.append(
            "optimizer_model_refits: "
            f"{timing['optimizer_model_refit_seconds']:.6f} "
            f"({timing['optimizer_model_refit_percent']:.2f}%) "
            "[subset of splitter]"
        )
    lines.append(
        (
            f"other: {timing['other_seconds']:.6f} "
            f"({timing['other_percent']:.2f}%)"
        )
    )
    return tuple(lines)


def add_splitter_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the splitter choice and parameters specific to HRT."""

    parser.add_argument(
        "--splitter",
        choices=("soft_oblique", "hrt"),
        default=TREE_DEFAULTS.splitter_name,
        help="Node splitter; HRT requires affine (degree-1) leaves.",
    )
    parser.add_argument(
        "--hrt-mode",
        choices=("max", "min", "both"),
        default=HRT_DEFAULTS.mode,
    )
    parser.add_argument("--hrt-mu", type=float, default=HRT_DEFAULTS.mu)
    parser.add_argument("--hrt-tol", type=float, default=HRT_DEFAULTS.tol)
    parser.add_argument(
        "--hrt-init-scale",
        type=float,
        default=HRT_DEFAULTS.init_scale,
    )


def add_tree_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared model, builder, sampler, temperature, and splitter options."""

    add_splitter_selection_arguments(parser)
    parser.add_argument("--traversal", choices=("dfs", "bfs"), default=TREE_DEFAULTS.traversal)
    parser.add_argument("--node-batch-size", type=int, default=TREE_DEFAULTS.node_batch_size)
    parser.add_argument("--batch-reorder", action=argparse.BooleanOptionalAction,
                        default=TREE_DEFAULTS.batch_reorder)
    parser.add_argument("--optimizer", choices=OPTIMIZERS,
                        default=SPLITTER_DEFAULTS.optimizer)
    parser.add_argument("--backtrack-batch-size", type=int, default=SPLITTER_DEFAULTS.backtrack_batch_size)
    parser.add_argument("--xw-refresh", dest="Xw_refresh", type=int, default=SPLITTER_DEFAULTS.Xw_refresh)
    parser.add_argument("--qr-parallel", action=argparse.BooleanOptionalAction,
                        default=SPLITTER_DEFAULTS.qr_parallel)
    parser.add_argument("--leaf-degree", type=int, default=TREE_DEFAULTS.leaf_degree)
    parser.add_argument(
        "--leaf-bias",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.leaf_include_bias,
        help="Include the constant feature in polynomial leaves.",
    )
    parser.add_argument(
        "--error-metric",
        choices=tuple(ERROR_METRICS),
        default=TREE_DEFAULTS.error_metric,
    )
    parser.add_argument(
        "--error-tolerance",
        "--error-threshold",
        dest="error_tolerance",
        type=float,
        default=TREE_DEFAULTS.error_tolerance,
    )
    parser.add_argument(
        "--relative-error-floor",
        type=float,
        default=TREE_DEFAULTS.relative_error_floor,
    )
    parser.add_argument("--max-depth", type=int, default=TREE_DEFAULTS.max_depth)
    parser.add_argument(
        "--sample-multiplier",
        type=int,
        default=TREE_DEFAULTS.sample_multiplier,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=TREE_DEFAULTS.sample_count,
        help="Fixed samples per node; overrides sample_multiplier*d_eff.",
    )
    parser.add_argument("--ridge", type=float, default=TREE_DEFAULTS.ridge)
    parser.add_argument(
        "--ridge-solver",
        choices=("auto", "normal", "qr", "svd"),
        default=TREE_DEFAULTS.ridge_solver,
    )
    parser.add_argument(
        "--auto-rcond-threshold",
        type=float,
        default=TREE_DEFAULTS.auto_rcond_threshold,
    )
    parser.add_argument(
        "--check-split-gain",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.check_split_gain,
        help="Require both absolute and relative split-gain thresholds (disabled by default).",
    )
    parser.add_argument(
        "--min-split-gain",
        type=float,
        default=TREE_DEFAULTS.min_split_gain,
    )
    parser.add_argument(
        "--min-relative-split-gain",
        type=float,
        default=TREE_DEFAULTS.min_relative_split_gain,
    )
    parser.add_argument(
        "--max-retries-on-failure",
        type=int,
        default=TREE_DEFAULTS.max_retries_on_failure,
    )
    parser.add_argument(
        "--min-side-fallback",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.min_side_fallback,
        help="Use the last feasible optimizer state after final min-side failure.",
    )
    parser.add_argument(
        "--validate",
        dest="validation_fraction",
        type=float,
        default=TREE_DEFAULTS.validation_fraction,
        metavar="FRACTION",
        help=(
            "Fit the convergence-check model on 1-FRACTION of node samples; "
            "supported only for soft splitting with max pointwise relative error."
        ),
    )

    parser.add_argument(
        "--sampler",
        choices=("hit_and_run", "parallel_numba"),
        default=SAMPLING_DEFAULTS.sampler_name,
    )
    parser.add_argument("--burn-in", type=int, default=SAMPLING_DEFAULTS.burn_in)
    parser.add_argument("--thinning", type=int, default=SAMPLING_DEFAULTS.thinning)
    parser.add_argument("--n-chains", type=int, default=SAMPLING_DEFAULTS.n_chains)
    parser.add_argument(
        "--sampler-threads",
        type=int,
        default=SAMPLING_DEFAULTS.threads,
        help="CPU threads used by parallel transitions and NumPy RNG.",
    )
    parser.add_argument(
        "--parallel-random",
        action=argparse.BooleanOptionalAction,
        default=SAMPLING_DEFAULTS.parallel_random,
        help="Generate chain-major normal and uniform arrays with NumPy threads.",
    )
    parser.add_argument(
        "--sampling-feasibility-tol",
        type=float,
        default=SAMPLING_DEFAULTS.feasibility_tol,
    )
    parser.add_argument(
        "--max-feasible-tries",
        type=int,
        default=SAMPLING_DEFAULTS.max_feasible_tries,
    )
    parser.add_argument(
        "--isotropic-sampling",
        action=argparse.BooleanOptionalAction,
        default=SAMPLING_DEFAULTS.isotropic_sampling,
    )
    parser.add_argument(
        "--isotropic-pilot-multiplier",
        type=int,
        default=SAMPLING_DEFAULTS.isotropic_pilot_multiplier,
    )
    parser.add_argument(
        "--direction-eigenvalue-floor",
        type=float,
        default=SAMPLING_DEFAULTS.direction_eigenvalue_floor,
    )
    parser.add_argument(
        "--exact-box-root",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.exact_box_root,
    )
    parser.add_argument(
        "--store-samples",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.store_samples,
    )
    parser.add_argument(
        "--store-leaf-representative",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.store_leaf_representative,
        help="Store the sampled-point centroid in each leaf.",
    )
    parser.add_argument(
        "--store-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.store_diagnostics,
    )
    parser.add_argument(
        "--profile-build-timing",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.profile_build_timing,
        help="Measure aggregate sampling, splitter, and optimizer-refit time.",
    )
    parser.add_argument(
        "--oracle-vectorized",
        action=argparse.BooleanOptionalAction,
        default=TREE_DEFAULTS.oracle_vectorized,
    )

    parser.add_argument(
        "--temperature-strategy",
        choices=("splitter", "fixed", "tune_root", "tune_node"),
        default=TEMPERATURE_DEFAULTS.strategy,
    )
    parser.add_argument(
        "--temperature-scale-mode",
        choices=("median_nn", "median_pairwise_scaled"),
        default=TEMPERATURE_DEFAULTS.scale_mode,
    )
    parser.add_argument(
        "--temperature-c-values",
        type=str,
        default=",".join(str(value) for value in TEMPERATURE_DEFAULTS.c_values),
    )
    parser.add_argument(
        "--temperature-validation-fraction",
        type=float,
        default=TEMPERATURE_DEFAULTS.validation_fraction,
    )
    parser.add_argument(
        "--max-temperature-points",
        type=int,
        default=TEMPERATURE_DEFAULTS.max_points,
    )
    parser.add_argument("--temperature-c", type=float, default=TEMPERATURE_DEFAULTS.c)
    parser.add_argument(
        "--nn-method",
        choices=("auto", "kdtree", "bruteforce"),
        default=TEMPERATURE_DEFAULTS.nn_method,
    )
    parser.add_argument(
        "--bruteforce-dimension-threshold",
        type=int,
        default=TEMPERATURE_DEFAULTS.bruteforce_dimension_threshold,
    )

    parser.add_argument(
        "--temperature-placeholder",
        type=float,
        default=SPLITTER_DEFAULTS.temperature_placeholder,
    )
    parser.add_argument("--max-iters", type=int, default=SPLITTER_DEFAULTS.max_iters)
    parser.add_argument("--grad-atol", type=float, default=SPLITTER_DEFAULTS.grad_atol)
    parser.add_argument("--grad-rtol", type=float, default=SPLITTER_DEFAULTS.grad_rtol)
    parser.add_argument(
        "--min-side-points",
        type=int,
        default=SPLITTER_DEFAULTS.min_side_points,
        help="Minimum hard-split side size, combined with min_side_fraction using max.",
    )
    parser.add_argument(
        "--min-side-fraction",
        type=float,
        default=SPLITTER_DEFAULTS.min_side_fraction,
    )
    parser.add_argument("--n-restarts", type=int, default=SPLITTER_DEFAULTS.n_restarts)
    parser.add_argument("--alpha0", type=float, default=SPLITTER_DEFAULTS.alpha0)
    parser.add_argument("--rho", type=float, default=SPLITTER_DEFAULTS.rho)
    parser.add_argument("--armijo-c", type=float, default=SPLITTER_DEFAULTS.armijo_c)
    parser.add_argument(
        "--max-backtracks",
        type=int,
        default=SPLITTER_DEFAULTS.max_backtracks,
    )
    parser.add_argument(
        "--adaptive-alpha",
        action=argparse.BooleanOptionalAction,
        default=SPLITTER_DEFAULTS.adaptive_alpha,
    )
    parser.add_argument("--alpha-min", type=float, default=SPLITTER_DEFAULTS.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=SPLITTER_DEFAULTS.alpha_max)
    parser.add_argument("--alpha-grow", type=float, default=SPLITTER_DEFAULTS.alpha_grow)
    parser.add_argument(
        "--alpha-recovery",
        type=float,
        default=SPLITTER_DEFAULTS.alpha_recovery,
    )
    parser.add_argument(
        "--heavy-backtrack-threshold",
        type=int,
        default=SPLITTER_DEFAULTS.heavy_backtrack_threshold,
    )
    parser.add_argument(
        "--max-line-search-failures",
        type=int,
        default=SPLITTER_DEFAULTS.max_line_search_failures,
    )
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=SPLITTER_DEFAULTS.weight_floor,
    )
    parser.add_argument(
        "--refit-during-line-search",
        action=argparse.BooleanOptionalAction,
        default=SPLITTER_DEFAULTS.refit_during_line_search,
    )


def configured_error_metric(
    name: str,
    relative_error_floor: float,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Resolve the selected stopping metric and relative-error floor."""

    metric = ERROR_METRICS[name]
    if metric is mean_squared_error:
        return metric
    configured = partial(metric, floor=relative_error_floor)
    configured.__name__ = metric.__name__
    return configured


def add_output_offset(
    target: Callable[[np.ndarray], float | np.ndarray],
    offset: float,
) -> Callable[[np.ndarray], float | np.ndarray]:
    """Return a target shifted by a finite constant output offset."""

    offset = float(offset)
    if not np.isfinite(offset):
        raise ValueError("offset must be finite.")
    if offset == 0.0:
        return target

    def shifted(x: np.ndarray) -> float | np.ndarray:
        return target(x) + offset

    return shifted


def make_benchmark_splitter(
    args: argparse.Namespace,
    model: object,
    min_side_points: int,
) -> tuple[Splitter, TemperatureConfig | None]:
    """Construct the selected splitter and its optional temperature policy."""

    if args.splitter == "hrt":
        if args.optimizer != "cpu":
            raise ValueError("HRT supports only --optimizer cpu.")
        if args.leaf_degree != 1:
            raise ValueError("--splitter hrt requires --leaf-degree 1.")
        splitter = HingeAffineSplitter(
            mode=args.hrt_mode,
            ridge=args.ridge,
            solver=args.ridge_solver,
            auto_rcond_threshold=args.auto_rcond_threshold,
            mu=args.hrt_mu,
            max_iters=args.max_iters,
            tol=args.hrt_tol,
            min_side_points=min_side_points,
            min_side_fraction=args.min_side_fraction,
            min_side_fallback=args.min_side_fallback,
            n_restarts=args.n_restarts,
            init_scale=args.hrt_init_scale,
            random_state=args.seed,
        )
        return splitter, None

    splitter = SoftObliqueSplitter(
        model_template=model.clone(),
        temperature=args.temperature_placeholder,
        max_iters=args.max_iters,
        grad_atol=args.grad_atol,
        grad_rtol=args.grad_rtol,
        min_side_points=min_side_points,
        min_side_fraction=args.min_side_fraction,
        min_side_fallback=args.min_side_fallback,
        n_restarts=args.n_restarts,
        alpha0=args.alpha0,
        rho=args.rho,
        armijo_c=args.armijo_c,
        max_backtracks=args.max_backtracks,
        adaptive_alpha=args.adaptive_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_grow=args.alpha_grow,
        alpha_recovery=args.alpha_recovery,
        heavy_backtrack_threshold=args.heavy_backtrack_threshold,
        max_line_search_failures=args.max_line_search_failures,
        weight_floor=args.weight_floor,
        refit_during_line_search=args.refit_during_line_search,
        optimizer=args.optimizer,
        backtrack_batch_size=args.backtrack_batch_size,
        Xw_refresh=args.Xw_refresh,
        qr_parallel=args.qr_parallel,
        random_state=args.seed,
    )
    temperature_config = TemperatureConfig(
        strategy=args.temperature_strategy,
        scale_mode=args.temperature_scale_mode,
        c=args.temperature_c,
        c_values=tuple(parse_csv_floats(args.temperature_c_values)),
        validation_fraction=args.temperature_validation_fraction,
        max_points=args.max_temperature_points,
        nn_method=None if args.nn_method == "auto" else args.nn_method,
        bruteforce_dimension_threshold=args.bruteforce_dimension_threshold,
    )
    return splitter, temperature_config


def build_tree_benchmark(
    args: argparse.Namespace,
    domain: BoxDomain,
    target: Callable[[np.ndarray], float | np.ndarray],
) -> TreeBenchmarkRun:
    """Construct and time a regression tree from shared benchmark arguments."""

    stopping_metric = configured_error_metric(
        args.error_metric,
        relative_error_floor=args.relative_error_floor,
    )
    leaf_include_bias = True if args.leaf_degree == 1 else args.leaf_bias
    model = make_model_template(
        args.leaf_degree,
        ridge=args.ridge,
        include_bias=leaf_include_bias,
        solver=args.ridge_solver,
        auto_rcond_threshold=args.auto_rcond_threshold,
    )
    min_side_points = args.min_side_points
    splitter, temperature_config = make_benchmark_splitter(
        args,
        model,
        min_side_points,
    )
    sampler, root_sampler = make_benchmark_samplers(args)
    builder = RegressionTreeBuilder(
        domain=domain,
        oracle=target,
        model_template=model,
        splitter=splitter,
        error_threshold=args.error_tolerance,
        max_depth=args.max_depth,
        error_metric=stopping_metric,
        sample_multiplier=args.sample_multiplier,
        sample_count=args.sample_count,
        sampler=sampler,
        root_sampler=root_sampler,
        exact_box_root=args.exact_box_root,
        isotropic_sampling=args.isotropic_sampling,
        isotropic_pilot_multiplier=args.isotropic_pilot_multiplier,
        temperature_config=temperature_config,
        min_split_gain=args.min_split_gain,
        min_relative_split_gain=args.min_relative_split_gain,
        check_split_gain=args.check_split_gain,
        max_retries_on_failure=args.max_retries_on_failure,
        validation_fraction=args.validation_fraction,
        store_samples=args.store_samples,
        store_leaf_representative=args.store_leaf_representative,
        store_diagnostics=args.store_diagnostics,
        profile_build_timing=args.profile_build_timing,
        oracle_vectorized=args.oracle_vectorized,
        random_state=args.seed,
        traversal=args.traversal,
        node_batch_size=args.node_batch_size,
        batch_reorder=args.batch_reorder,
    )

    start = time.perf_counter()
    build_result = builder.build()
    build_seconds = time.perf_counter() - start
    return TreeBenchmarkRun(
        builder=builder,
        build_result=build_result,
        tree=build_result.tree,
        model=model,
        leaf_include_bias=leaf_include_bias,
        min_side_points=min_side_points,
        build_seconds=build_seconds,
    )


def make_benchmark_samplers(
    args: argparse.Namespace,
) -> tuple[HitAndRunSampler, HitAndRunSampler]:
    """Construct matching node and root samplers from shared CLI settings."""

    common = {
        "burn_in": args.burn_in,
        "thinning": args.thinning,
        "feasibility_tol": args.sampling_feasibility_tol,
        "max_feasible_tries": args.max_feasible_tries,
        "direction_eigenvalue_floor": args.direction_eigenvalue_floor,
    }
    if args.sampler == "hit_and_run":
        return HitAndRunSampler(**common), HitAndRunSampler(**common)
    if args.sampler != "parallel_numba":
        raise ValueError(f"Unknown sampler {args.sampler!r}.")
    if args.n_chains < 1:
        raise ValueError("--n-chains must be at least 1.")
    if args.sampler_threads < 1:
        raise ValueError("--sampler-threads must be at least 1.")

    from numba import config, set_num_threads

    resolved_threads = min(args.sampler_threads, config.NUMBA_NUM_THREADS)
    set_num_threads(resolved_threads)
    parallel = {
        **common,
        "n_chains": args.n_chains,
        "parallel_random": args.parallel_random,
        "random_threads": resolved_threads,
    }
    return (
        ParallelNumbaBatchedHitAndRunSampler(**parallel),
        ParallelNumbaBatchedHitAndRunSampler(**parallel),
    )


def pointwise_test_errors(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relative_error_floor: float,
) -> tuple[np.ndarray, str]:
    """Return point-level errors whose aggregate matches the selected metric."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    residual = np.abs(y_true - y_pred)
    if metric_name == "mse":
        return residual**2, "squared error"
    if metric_name in ("median_pointwise_relative", "max_pointwise_relative"):
        return (
            pointwise_relative_error(
                y_true,
                y_pred,
                floor=relative_error_floor,
            ),
            "pointwise relative error",
        )
    if metric_name == "relative_l2":
        denominator = max(float(np.linalg.norm(y_true)), float(relative_error_floor))
        return (
            np.sqrt(y_true.size) * residual / denominator,
            "normalized relative L2 contribution",
        )
    raise ValueError(f"Unknown error metric {metric_name!r}.")


def evaluate_tree_benchmark(
    tree: RegressionTree,
    target: Callable[[np.ndarray], float | np.ndarray],
    domain: BoxDomain,
    n_test: int,
    error_metric: str,
    relative_error_floor: float,
    random_state: int | np.random.Generator | None,
    batch_size: int | None = None,
    point_sampler: Callable[[int, np.random.Generator], np.ndarray] | None = None,
) -> TreeTestEvaluation:
    """Evaluate a tree on independent test points, uniform by default."""

    if n_test < 1:
        raise ValueError("n_test must be positive.")
    if batch_size is None:
        batch_size = n_test
    if batch_size < 1:
        raise ValueError("batch_size must be positive or None.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    y_true = np.empty(n_test, dtype=float)
    y_pred = np.empty(n_test, dtype=float)
    bad_node_counts = Counter() if point_sampler is None else None
    test_sampling_seconds = 0.0
    test_oracle_seconds = 0.0
    prediction_seconds = 0.0
    for start_index in range(0, n_test, batch_size):
        stop_index = min(start_index + batch_size, n_test)
        start = time.perf_counter()
        batch_count = stop_index - start_index
        X_batch = (
            sample_uniform_box(domain.bounds, batch_count, random_state=rng)
            if point_sampler is None
            else np.asarray(point_sampler(batch_count, rng), dtype=float)
        )
        if X_batch.shape != (batch_count, domain.dimension):
            raise ValueError(
                "point_sampler must return shape "
                f"{(batch_count, domain.dimension)}, got {X_batch.shape}."
            )
        test_sampling_seconds += time.perf_counter() - start
        start = time.perf_counter()
        y_true[start_index:stop_index] = np.asarray(
            target(X_batch),
            dtype=float,
        ).reshape(-1)
        test_oracle_seconds += time.perf_counter() - start
        start = time.perf_counter()
        y_pred[start_index:stop_index] = tree.predict(X_batch)
        prediction_seconds += time.perf_counter() - start
        if bad_node_counts is not None:
            for condition, mask in _bad_node_membership_masks(tree, X_batch).items():
                bad_node_counts[condition] += int(np.count_nonzero(mask))

    relative_quantiles = np.asarray(
        pointwise_relative_error_quantile(
            y_true,
            y_pred,
            POINTWISE_QUANTILES,
            floor=relative_error_floor,
        ),
        dtype=float,
    )
    selected_errors, selected_label = pointwise_test_errors(
        error_metric,
        y_true,
        y_pred,
        relative_error_floor,
    )
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_relative_l2_error": relative_l2_error(
            y_true,
            y_pred,
            floor=relative_error_floor,
        ),
        "test_median_pointwise_relative_error": median_pointwise_relative_error(
            y_true,
            y_pred,
            floor=relative_error_floor,
        ),
        "test_max_pointwise_relative_error": max_pointwise_relative_error(
            y_true,
            y_pred,
            floor=relative_error_floor,
        ),
        "test_max_absolute_error": float(np.max(np.abs(y_true - y_pred))),
    }
    return TreeTestEvaluation(
        metrics=metrics,
        pointwise_relative_quantiles=dict(zip(POINTWISE_QUANTILES, relative_quantiles)),
        selected_point_errors=selected_errors,
        selected_point_error_label=selected_label,
        oracle_min=float(np.min(y_true)),
        oracle_max=float(np.max(y_true)),
        timings={
            "test_sampling": test_sampling_seconds,
            "test_oracle_evaluation": test_oracle_seconds,
            "tree_prediction": prediction_seconds,
            "predictions_per_second": n_test / max(prediction_seconds, 1e-12),
        },
        bad_node_test_fractions=(
            None
            if bad_node_counts is None
            else {
                condition: bad_node_counts[condition] / n_test
                for condition in (
                    "min_side_points",
                    "small_init_grad",
                    "max_depth",
                    "combined",
                )
            }
        ),
    )


def _bad_node_membership_masks(
    tree: RegressionTree,
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mark points lying in explicitly flagged bad-node regions."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}.")
    if tree.root is None:
        raise ValueError("Tree has no root node.")

    min_side = np.zeros(X.shape[0], dtype=bool)
    small_init_grad = np.zeros(X.shape[0], dtype=bool)
    max_depth = np.zeros(X.shape[0], dtype=bool)
    pending = [(tree.root, np.arange(X.shape[0]))]

    while pending:
        node, indices = pending.pop()
        if node.status == "min_side_points":
            min_side[indices] = True
        if "small_init_grad" in node.metadata.get("warnings", ()):
            small_init_grad[indices] = True
        if node.status == "max_depth":
            max_depth[indices] = True

        if isinstance(node, SplitNode):
            right_mask = X[indices] @ node.w - node.z >= 0.0
            if np.any(~right_mask):
                pending.append((node.left, indices[~right_mask]))
            if np.any(right_mask):
                pending.append((node.right, indices[right_mask]))

    return {
        "min_side_points": min_side,
        "small_init_grad": small_init_grad,
        "max_depth": max_depth,
        "combined": min_side | small_init_grad | max_depth,
    }


def save_tree_artifact(
    run: TreeBenchmarkRun,
    path: Path,
    args: argparse.Namespace,
    domain: BoxDomain,
    target_metadata: dict[str, object],
) -> float:
    """Save a tree and its resolved run configuration; return elapsed seconds."""

    run_config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_domain_bounds": domain.bounds.tolist(),
        "resolved_min_side_points": run.min_side_points,
        "target": target_metadata,
    }
    start = time.perf_counter()
    run.tree.save(path, run_config=run_config)
    elapsed = time.perf_counter() - start
    print(f"saved tree: {path}")
    return elapsed


def write_node_diagnostics(
    tree: RegressionTree,
    path: Path,
    extra_columns: Callable[[TreeNode], Mapping[str, object]] | None = None,
) -> None:
    """Write one compact diagnostics row per tree node."""

    rows = []
    for node in tree.iter_nodes():
        row = {
            "node_id": node.node_id,
            "node_type": "leaf" if isinstance(node, LeafNode) else "split",
            "depth": node.depth,
            "status": node.status,
            "fit_error": node.metadata.get("fit_error"),
            "fit_mse": node.metadata.get("fit_mse"),
            "convergence_error": node.metadata.get("convergence_error"),
            "convergence_fit_samples": node.metadata.get(
                "convergence_fit_samples"
            ),
            "fit_solver_requested": node.metadata.get("fit_solver_requested"),
            "fit_solver_used": node.metadata.get("fit_solver_used"),
            "fit_condition_estimator": node.metadata.get("fit_condition_estimator"),
            "fit_cond_estimate": node.metadata.get("fit_cond_estimate"),
            "fit_fallback_reason": node.metadata.get("fit_fallback_reason"),
            "n_samples": node.metadata.get("n_samples"),
            "n_inherited": node.metadata.get("n_inherited"),
            "n_new": node.metadata.get("n_new"),
            "sampling_method": node.metadata.get("sampling_method"),
            "sampling_thinning": node.metadata.get("sampling_thinning"),
            "sampling_pilot_length": node.metadata.get("sampling_pilot_length"),
            "sampling_covariance_condition_number": node.metadata.get(
                "sampling_covariance_condition_number"
            ),
            "sampling_covariance_floor_saturated": node.metadata.get(
                "sampling_covariance_floor_saturated"
            ),
            "sampling_warnings": ";".join(
                node.metadata.get("sampling_warnings", ())
            ),
            "split_gain_mse": node.metadata.get("split_gain_mse"),
            "relative_split_gain_mse": node.metadata.get(
                "relative_split_gain_mse"
            ),
            "split_stop_reason": node.metadata.get("split_stop_reason"),
            "split_iterations": node.metadata.get("split_iterations"),
            "temperature_c": node.metadata.get("temperature_c"),
            "temperature": node.metadata.get("temperature"),
            "restarts_on_failure": node.metadata.get("restarts_on_failure"),
            "split_attempt_failure_reasons": ";".join(
                node.metadata.get("split_attempt_failure_reasons", ())
            ),
            "warnings": ";".join(node.metadata.get("warnings", ())),
        }
        if extra_columns is not None:
            additions = dict(extra_columns(node))
            overlap = row.keys() & additions.keys()
            if overlap:
                raise ValueError(
                    "extra diagnostic columns cannot replace shared columns: "
                    f"{sorted(overlap)}"
                )
            row.update(additions)
        rows.append(row)
    if not rows:
        raise ValueError("Cannot write diagnostics for an empty tree.")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved diagnostics: {path}")


def write_benchmark_report(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    domain: BoxDomain,
    target_metadata: dict[str, object],
    run: TreeBenchmarkRun,
    evaluation: TreeTestEvaluation,
    tree_path: Path,
    tree_save_seconds: float,
    extra_timings: dict[str, float] | None = None,
    extra_configuration: Mapping[str, object] | None = None,
) -> None:
    """Write the common benchmark configuration, accuracy, and tree summary."""

    tree = run.tree
    nodes = list(tree.iter_nodes())
    leaves = [node for node in nodes if isinstance(node, LeafNode)]
    split_nodes = [node for node in nodes if isinstance(node, SplitNode)]
    leaf_statuses = Counter(node.status for node in leaves)
    split_statuses = Counter(node.status for node in split_nodes)
    optimizer_stop_reasons = Counter(
        str(node.metadata.get("split_stop_reason", "unknown")) for node in split_nodes
    )
    split_warnings = Counter(
        warning
        for node in split_nodes
        for warning in node.metadata.get("warnings", ())
    )
    node_warnings = Counter(
        warning for node in nodes for warning in node.metadata.get("warnings", ())
    )
    sampling_warnings = Counter(
        warning
        for node in nodes
        for warning in node.metadata.get("sampling_warnings", ())
    )
    leaf_errors = np.asarray(
        [leaf.metadata.get("fit_error", np.nan) for leaf in leaves],
        dtype=float,
    )
    min_side_depths = [
        node.depth for node in leaves if node.status == "min_side_points"
    ]
    small_init_grad_depths = [
        node.depth
        for node in nodes
        if "small_init_grad" in node.metadata.get("warnings", ())
    ]
    bad_node_fractions = evaluation.bad_node_test_fractions or {
        "min_side_points": 0.0,
        "small_init_grad": 0.0,
        "max_depth": 0.0,
        "combined": 0.0,
    }
    timings = {
        "tree_build": run.build_seconds,
        "tree_save": tree_save_seconds,
        **evaluation.timings,
        **({} if extra_timings is None else extra_timings),
    }

    lines = [
        "[configuration]",
        *(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        f"resolved_output_dir: {output_dir}",
        f"resolved_domain_bounds: {domain.bounds.tolist()}",
        f"dimension: {domain.dimension}",
        f"leaf_model: {type(run.model).__name__}",
        f"resolved_splitter: {type(run.builder.splitter).__name__}",
        f"resolved_hard_refits: {run.builder.hard_refits}",
        f"resolved_sampler: {type(run.builder.sampler).__name__}",
        f"resolved_sampler_n_chains: {getattr(run.builder.sampler, 'n_chains', None)}",
        f"resolved_sampler_threads: {getattr(run.builder.sampler, 'random_threads', None)}",
        f"resolved_parallel_random: {getattr(run.builder.sampler, 'parallel_random', None)}",
        f"leaf_include_bias: {run.leaf_include_bias}",
        f"effective_dimension: {tree.metadata.get('effective_dimension')}",
        f"resolved_min_side_points: {run.min_side_points}",
        f"target_samples_per_node: {run.builder.target_samples}",
        f"tree_artifact: {tree_path}",
        f"resolved_temperature_strategy: {tree.metadata.get('temperature_strategy')}",
        *(
            ()
            if extra_configuration is None
            else tuple(
                f"{key}: {value}" for key, value in extra_configuration.items()
            )
        ),
        "",
        "[target]",
        *(f"{key}: {value}" for key, value in target_metadata.items()),
        "",
        "[performance]",
        *(f"{key}: {value:.12e}" for key, value in evaluation.metrics.items()),
        f"test_error_histogram_quantity: {evaluation.selected_point_error_label}",
        "",
        "[test_oracle_range]",
        f"sample_count: {args.n_test}",
        f"estimated_min: {evaluation.oracle_min:.12e}",
        f"estimated_max: {evaluation.oracle_max:.12e}",
        f"estimated_span: {evaluation.oracle_max - evaluation.oracle_min:.12e}",
        "",
        "[pointwise_relative_error_quantiles]",
        *(
            f"q{100 * quantile:g}: {value:.12e}"
            for quantile, value in evaluation.pointwise_relative_quantiles.items()
        ),
        "",
        "[tree]",
        f"oracle_queries: {run.build_result.oracle_queries}",
        f"restarts_on_failure: {run.build_result.restarts_on_failure}",
        f"num_nodes: {tree.num_nodes()}",
        f"num_split_nodes: {tree.num_split_nodes()}",
        f"num_leaves: {tree.num_leaves()}",
        f"realized_max_depth: {tree.max_depth()}",
        f"min_leaf_fit_error: {np.nanmin(leaf_errors):.12e}",
        f"max_leaf_fit_error: {np.nanmax(leaf_errors):.12e}",
        f"selected_root_temperature_c: {tree.root.metadata.get('temperature_c')}",
        f"selected_root_temperature: {tree.root.metadata.get('temperature')}",
        "",
        "[timing_seconds]",
        *(f"{key}: {value:.6f}" for key, value in timings.items()),
        "",
        "[build_timing_profile]",
        *build_timing_report_lines(
            run.build_result.build_timing,
            include_model_refits=args.splitter != "hrt",
        ),
        "",
        "[leaf_status_counts]",
        *(
            f"{status}: {leaf_statuses.get(status, 0)}"
            for status in EXPECTED_LEAF_STATUSES
        ),
        *(
            f"{status}: {count}"
            for status, count in sorted(leaf_statuses.items())
            if status not in EXPECTED_LEAF_STATUSES
        ),
        "",
        "[split_status_counts]",
        *(f"{status}: {count}" for status, count in sorted(split_statuses.items())),
        "",
        "[optimizer_stop_reason_counts]",
        *(
            f"{reason}: {count}"
            for reason, count in sorted(optimizer_stop_reasons.items())
        ),
        "",
        "[split_warning_counts]",
        *(f"{warning}: {count}" for warning, count in sorted(split_warnings.items())),
        "",
        "[node_warning_counts]",
        *(f"{warning}: {count}" for warning, count in sorted(node_warnings.items())),
        "",
        "[bad_node_volume]",
        (
            "min_side_points_earliest_depth: "
            f"{min(min_side_depths) if min_side_depths else None}"
        ),
        (
            "min_side_points_test_fraction: "
            f"{bad_node_fractions['min_side_points']:.12e}"
        ),
        (
            "small_init_grad_earliest_depth: "
            f"{min(small_init_grad_depths) if small_init_grad_depths else None}"
        ),
        (
            "small_init_grad_test_fraction: "
            f"{bad_node_fractions['small_init_grad']:.12e}"
        ),
        f"max_depth_test_fraction: {bad_node_fractions['max_depth']:.12e}",
        f"combined_test_fraction: {bad_node_fractions['combined']:.12e}",
        "",
        "[sampling_warnings]",
        (
            "covariance_eigenvalue_floor_saturated: "
            f"{sampling_warnings.get('covariance_eigenvalue_floor_saturated', 0)}"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"saved report: {path}")
