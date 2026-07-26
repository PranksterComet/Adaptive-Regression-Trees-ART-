"""Build and evaluate a 2D adaptive regression tree on benchmark targets."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import partial
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.builder import RegressionTreeBuilder
from art.domain import BoxDomain
from art.metrics import (
    max_pointwise_relative_error,
    mean_squared_error,
    median_pointwise_relative_error,
    relative_l2_error,
)
from art.sampling import HitAndRunSampler, sample_uniform_box
from art.splitters import SoftObliqueSplitter
from art.temperature import DEFAULT_TEMPERATURE_GRID, TemperatureConfig
from art.tree import LeafNode, SplitNode

from examples.benchmark_functions import (
    GAUSSIAN_DEFAULT_INTERVAL,
    PLANE_WAVE_DEFAULT_INTERVAL,
    QUADRATIC_DEFAULT_INTERVAL,
    ROSENBROCK_DEFAULT_INTERVAL,
    SPHERICAL_PIECEWISE_DEFAULT_INTERVAL,
    GaussianFunction,
    PlaneWaveFunction,
    QuadraticFunction,
    RosenbrockFunction,
    SphericalPiecewisePolynomialFunction,
    default_gaussian_mixture_2d,
    random_orthogonal_matrix,
    rotation_matrix_2d,
)
from examples.plotting_helpers import (
    make_tree_leaf_grid,
    save_function_contour,
    save_tree_leaf_error_regions,
    save_tree_leaf_regions,
)
from examples.test_helpers import make_model_template, parse_csv_floats


DIMENSION = 2
DEFAULT_OUTPUT_ROOT = Path(__file__).with_name("tree_2D_benchmark_outputs")
BENCHMARK_DEFAULT_INTERVALS = {
    "quadratic": QUADRATIC_DEFAULT_INTERVAL,
    "gaussian": GAUSSIAN_DEFAULT_INTERVAL,
    "gaussian_mixture": GAUSSIAN_DEFAULT_INTERVAL,
    "plane_wave": PLANE_WAVE_DEFAULT_INTERVAL,
    "spherical_piecewise": SPHERICAL_PIECEWISE_DEFAULT_INTERVAL,
    "rosenbrock": ROSENBROCK_DEFAULT_INTERVAL,
}
ERROR_METRICS = {
    "relative_l2": relative_l2_error,
    "mse": mean_squared_error,
    "median_pointwise_relative": median_pointwise_relative_error,
    "max_pointwise_relative": max_pointwise_relative_error,
}


def configured_error_metric(
    name: str,
    relative_error_floor: float,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Resolve a named stopping metric, including its relative-error floor."""

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
    """Return a benchmark shifted by a constant output offset."""

    offset = float(offset)
    if not np.isfinite(offset):
        raise ValueError("offset must be finite.")
    if offset == 0.0:
        return target

    def shifted(x: np.ndarray) -> float | np.ndarray:
        return target(x) + offset

    return shifted


def random_polynomial_piece(
    dimension: int,
    degree: int,
    rng: np.random.Generator,
) -> QuadraticFunction:
    """Generate a reproducible affine or quadratic benchmark piece."""

    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2.")
    Q = np.eye(dimension)
    eigenvalues = np.zeros(dimension)
    if degree == 2:
        Q = random_orthogonal_matrix(dimension, random_state=rng)
        eigenvalues = rng.normal(scale=0.25, size=dimension)
    return QuadraticFunction(
        dimension=dimension,
        beta=float(rng.normal()),
        w=rng.normal(scale=dimension**-0.5, size=dimension),
        Q=Q,
        Lambda=np.diag(eigenvalues),
    )


def make_benchmark_target(
    args: argparse.Namespace,
) -> tuple[Callable[[np.ndarray], float | np.ndarray], dict[str, object]]:
    """Construct the selected 2D benchmark and its report metadata."""

    spectrum = np.asarray(args.spectrum, dtype=float)
    rotation = rotation_matrix_2d(args.rotation_angle_degrees, degrees=True)
    if args.benchmark == "quadratic":
        target = QuadraticFunction(
            dimension=DIMENSION,
            beta=0.0,
            w=np.asarray(args.quadratic_linear, dtype=float),
            Q=rotation,
            Lambda=np.diag(spectrum),
        )
        metadata = {
            "beta": target.beta,
            "w": target.w.tolist(),
            "Q": target.Q.tolist(),
            "Lambda_diagonal": np.diag(target.Lambda).tolist(),
        }
    elif args.benchmark == "gaussian":
        target = GaussianFunction(
            dimension=DIMENSION,
            beta=0.0,
            Q=rotation,
            Sigma=np.diag(spectrum),
        )
        metadata = {
            "beta": target.beta,
            "Q": target.Q.tolist(),
            "Sigma_diagonal": np.diag(target.Sigma).tolist(),
        }
    elif args.benchmark == "gaussian_mixture":
        target = default_gaussian_mixture_2d(beta=0.0)
        metadata = {
            "beta": target.beta,
            "weights": target.weights.tolist(),
            "component_means": [
                component.mean.tolist() for component in target.components
            ],
            "component_covariances": [
                component.covariance.tolist() for component in target.components
            ],
        }
    elif args.benchmark == "plane_wave":
        random_features = args.plane_wave_random_features
        target = PlaneWaveFunction(
            dimension=DIMENSION,
            beta=0.0,
            frequency=args.plane_wave_frequency,
            normal=(
                None
                if random_features
                else np.asarray(args.plane_wave_normal, dtype=float)
            ),
            phase=None if random_features else args.plane_wave_phase,
            amplitude=args.plane_wave_amplitude,
            feature_mode="random" if random_features else "explicit",
            random_state=args.seed,
        )
        metadata = {
            "beta": target.beta,
            "amplitude": target.amplitude,
            "frequency": target.frequency,
            "normal": target.normal.tolist(),
            "phase": target.phase,
            "feature_mode": target.feature_mode,
        }
    elif args.benchmark == "spherical_piecewise":
        rng = np.random.default_rng(args.seed)
        inside = random_polynomial_piece(
            DIMENSION,
            args.sphere_piece_degree,
            rng,
        )
        outside = random_polynomial_piece(
            DIMENSION,
            args.sphere_piece_degree,
            rng,
        )
        target = SphericalPiecewisePolynomialFunction(
            inside_polynomial=inside,
            outside_polynomial=outside,
            radius=args.sphere_radius,
        )
        metadata = {
            "radius": target.radius,
            "center": target.center.tolist(),
            "piece_degree": args.sphere_piece_degree,
            "inside_beta": inside.beta,
            "inside_w": inside.w.tolist(),
            "inside_quadratic_matrix": inside.quadratic_matrix.tolist(),
            "outside_beta": outside.beta,
            "outside_w": outside.w.tolist(),
            "outside_quadratic_matrix": outside.quadratic_matrix.tolist(),
        }
    else:
        target = RosenbrockFunction(
            dimension=DIMENSION,
            a=args.rosenbrock_a,
            b=args.rosenbrock_b,
        )
        metadata = {"a": target.a, "b": target.b}
    metadata["offset"] = float(args.offset)
    return add_output_offset(target, args.offset), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Experiment and target settings.
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_DEFAULT_INTERVALS),
        default="quadratic",
    )
    parser.add_argument("--seed", type=int, default=7, help="Tree construction seed.")
    parser.add_argument("--test-seed", type=int, default=19, help="Independent test sample seed.")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Constant added to the selected benchmark output.",
    )
    parser.add_argument("--n-test", type=int, default=DIMENSION * 10_000)
    parser.add_argument("--domain-low", type=float, default=None)
    parser.add_argument("--domain-high", type=float, default=None)
    parser.add_argument("--rotation-angle-degrees", type=float, default=15.0)
    parser.add_argument("--spectrum", type=float, nargs=2, default=(1.0, 1.0 / 8.0))
    parser.add_argument("--quadratic-linear", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--plane-wave-amplitude", type=float, default=1.0)
    parser.add_argument(
        "--plane-wave-frequency",
        type=float,
        default=4.0,
        help="Angular frequency in radians per unit distance along the normal.",
    )
    parser.add_argument(
        "--plane-wave-normal",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        help="Plane normal; normalized internally.",
    )
    parser.add_argument(
        "--plane-wave-phase",
        type=float,
        default=0.0,
        help="Phase shift in radians.",
    )
    parser.add_argument(
        "--plane-wave-random-features",
        action="store_true",
        help="Draw a unit normal and phase from the tree-construction seed.",
    )
    parser.add_argument("--sphere-radius", type=float, default=2.0)
    parser.add_argument(
        "--sphere-piece-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help="Polynomial degree of the randomly generated inside/outside pieces.",
    )
    parser.add_argument("--rosenbrock-a", type=float, default=1.0)
    parser.add_argument("--rosenbrock-b", type=float, default=100.0)
    parser.add_argument("--grid-resolution", type=int, default=500)
    parser.add_argument("--contour-levels", type=int, default=30)
    parser.add_argument(
        "--contour-scale",
        choices=("linear", "log", "symlog", "auto"),
        default="auto",
    )
    parser.add_argument("--contour-dynamic-range-threshold", type=float, default=1e3)
    parser.add_argument("--symlog-linthresh", type=float, default=None)
    parser.add_argument(
        "--label-leaves",
        action="store_true",
        help="Annotate leaf regions with labels matching node_diagnostics.csv.",
    )
    parser.add_argument("--relative-error-floor", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Tree and polynomial leaf-model settings.
    parser.add_argument("--leaf-degree", type=int, default=1)
    parser.add_argument(
        "--no-leaf-bias",
        action="store_true",
        help="Exclude the constant feature from polynomial leaves of degree greater than one.",
    )
    parser.add_argument(
        "--error-metric",
        choices=tuple(ERROR_METRICS),
        default="relative_l2",
        help="Training error metric used to decide whether a node becomes a leaf.",
    )
    parser.add_argument(
        "--error-tolerance",
        "--error-threshold",
        dest="error_tolerance",
        type=float,
        default=1e-2,
        help="Stop splitting a node when its selected training error is at most this value.",
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--sample-multiplier", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--min-split-gain", type=float, default=0.0)
    parser.add_argument("--min-relative-split-gain", type=float, default=1e-3)

    # Hit-and-run settings for topping up child samples.
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--thinning", type=int, default=20)

    # Root-only temperature tuning settings.
    parser.add_argument(
        "--temperature-scale-mode",
        choices=("median_nn", "median_pairwise_scaled"),
        default="median_nn",
    )
    parser.add_argument(
        "--temperature-c-values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_TEMPERATURE_GRID),
    )
    parser.add_argument("--temperature-validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-temperature-points", type=int, default=512)
    parser.add_argument("--temperature-c", type=float, default=0.1)
    parser.add_argument("--nn-method", choices=("auto", "kdtree", "bruteforce"), default="auto")
    parser.add_argument("--bruteforce-dimension-threshold", type=int, default=20)

    # Soft-oblique optimizer settings, matching the stress-test defaults.
    parser.add_argument("--temperature-placeholder", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--grad-atol", type=float, default=1e-8)
    parser.add_argument("--grad-rtol", type=float, default=1e-5)
    parser.add_argument("--min-side-points", type=int, default=8)
    parser.add_argument("--min-side-fraction", type=float, default=0.00)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument(
        "--max-retries-on-failure",
        type=int,
        default=0,
        help="Additional splitter attempts after min-side or split-gain rejection.",
    )
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--armijo-c", type=float, default=1e-4)
    parser.add_argument("--max-backtracks", type=int, default=25)
    parser.add_argument(
        "--adaptive-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--alpha-min", type=float, default=1e-12)
    parser.add_argument("--alpha-max", type=float, default=1e8)
    parser.add_argument("--alpha-grow", type=float, default=10.0)
    parser.add_argument("--alpha-recovery", type=float, default=10.0)
    parser.add_argument("--heavy-backtrack-threshold", type=int, default=8)
    parser.add_argument("--max-line-search-failures", type=int, default=5)
    parser.add_argument("--weight-floor", type=float, default=1e-12)
    parser.add_argument(
        "--refit-during-line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit weighted models for every Armijo candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_interval = BENCHMARK_DEFAULT_INTERVALS[args.benchmark]
    domain_low = default_interval[0] if args.domain_low is None else args.domain_low
    domain_high = default_interval[1] if args.domain_high is None else args.domain_high
    if domain_low >= domain_high:
        raise ValueError("--domain-low must be less than --domain-high.")
    if args.benchmark == "spherical_piecewise":
        center = np.zeros(DIMENSION)
        low = np.full(DIMENSION, domain_low)
        high = np.full(DIMENSION, domain_high)
        min_distance = float(np.linalg.norm(center - np.clip(center, low, high)))
        max_distance = float(
            np.linalg.norm(np.maximum(np.abs(low - center), np.abs(high - center)))
        )
        if not min_distance < args.sphere_radius < max_distance:
            raise ValueError(
                "--sphere-radius must place part of the domain on each side of the sphere."
            )
    if args.n_test < 1:
        raise ValueError("--n-test must be positive.")
    if args.leaf_degree < 1:
        raise ValueError("--leaf-degree must be at least 1.")
    if args.grid_resolution < 2:
        raise ValueError("--grid-resolution must be at least 2.")
    if args.contour_levels < 2:
        raise ValueError("--contour-levels must be at least 2.")
    if args.contour_dynamic_range_threshold <= 1.0:
        raise ValueError("--contour-dynamic-range-threshold must be greater than 1.")
    if args.symlog_linthresh is not None and args.symlog_linthresh <= 0.0:
        raise ValueError("--symlog-linthresh must be positive.")
    if args.relative_error_floor <= 0.0:
        raise ValueError("--relative-error-floor must be positive.")
    if not np.all(np.isfinite(args.spectrum)):
        raise ValueError("--spectrum values must be finite.")
    if args.benchmark == "gaussian" and np.any(np.asarray(args.spectrum) <= 0.0):
        raise ValueError("Gaussian --spectrum values must be positive.")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.benchmark / f"degree_{args.leaf_degree}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    domain = BoxDomain.hypercube(DIMENSION, domain_low, domain_high)
    stopping_metric = configured_error_metric(
        args.error_metric,
        relative_error_floor=args.relative_error_floor,
    )

    target, target_metadata = make_benchmark_target(args)
    leaf_include_bias = True if args.leaf_degree == 1 else not args.no_leaf_bias
    model = make_model_template(
        args.leaf_degree,
        ridge=args.ridge,
        include_bias=leaf_include_bias,
    )
    splitter = SoftObliqueSplitter(
        model_template=model.clone(),
        # The builder replaces this value using root-only temperature tuning.
        temperature=args.temperature_placeholder,
        max_iters=args.max_iters,
        grad_atol=args.grad_atol,
        grad_rtol=args.grad_rtol,
        min_side_points=args.min_side_points,
        min_side_fraction=args.min_side_fraction,
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
        random_state=args.seed,
    )
    temperature_config = TemperatureConfig(
        strategy="tune_root",
        scale_mode=args.temperature_scale_mode,
        c=args.temperature_c,
        c_values=tuple(parse_csv_floats(args.temperature_c_values)),
        validation_fraction=args.temperature_validation_fraction,
        max_points=args.max_temperature_points,
        nn_method=None if args.nn_method == "auto" else args.nn_method,
        bruteforce_dimension_threshold=args.bruteforce_dimension_threshold,
    )
    sampler = HitAndRunSampler(burn_in=args.burn_in, thinning=args.thinning)
    builder = RegressionTreeBuilder(
        domain=domain,
        oracle=target,
        model_template=model,
        splitter=splitter,
        error_threshold=args.error_tolerance,
        max_depth=args.max_depth,
        error_metric=stopping_metric,
        sample_multiplier=args.sample_multiplier,
        sample_count=None,
        sampler=sampler,
        root_sampler=HitAndRunSampler(burn_in=args.burn_in, thinning=args.thinning),
        exact_box_root=True,
        temperature_config=temperature_config,
        min_split_gain=args.min_split_gain,
        min_relative_split_gain=args.min_relative_split_gain,
        max_retries_on_failure=args.max_retries_on_failure,
        store_samples=False,
        store_diagnostics=True,
        oracle_vectorized=True,
        random_state=args.seed,
    )

    start = time.perf_counter()
    build_result = builder.build()
    build_seconds = time.perf_counter() - start
    tree = build_result.tree

    start = time.perf_counter()
    X_test = sample_uniform_box(domain.bounds, args.n_test, random_state=args.test_seed)
    test_sampling_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_test = np.asarray(target(X_test), dtype=float)
    test_oracle_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = tree.predict(X_test)
    prediction_seconds = time.perf_counter() - start

    test_mse = mean_squared_error(y_test, y_pred)
    test_relative_l2 = relative_l2_error(y_test, y_pred, floor=args.relative_error_floor)
    test_median_relative = median_pointwise_relative_error(
        y_test, y_pred, floor=args.relative_error_floor
    )
    test_max_relative = max_pointwise_relative_error(
        y_test, y_pred, floor=args.relative_error_floor
    )
    test_max_absolute = float(np.max(np.abs(y_test - y_pred)))

    nodes = list(tree.iter_nodes())
    leaves = [node for node in nodes if isinstance(node, LeafNode)]
    split_nodes = [node for node in nodes if isinstance(node, SplitNode)]
    leaf_plot_labels = {id(leaf): f"L{index}" for index, leaf in enumerate(leaves)}
    leaf_statuses = Counter(leaf.status for leaf in leaves)
    split_statuses = Counter(node.status for node in split_nodes)
    optimizer_stop_reasons = Counter(
        str(node.metadata.get("split_stop_reason", "unknown")) for node in split_nodes
    )
    split_warnings = Counter(
        warning
        for node in split_nodes
        for warning in node.metadata.get("warnings", ())
    )

    plot_start = time.perf_counter()
    resolved_contour_scale = save_function_contour(
        target,
        domain.bounds,
        title=f"{args.benchmark.replace('_', ' ').title()} benchmark function",
        out_path=output_dir / "benchmark_contour.png",
        resolution=args.grid_resolution,
        levels=args.contour_levels,
        scale=args.contour_scale,
        dynamic_range_threshold=args.contour_dynamic_range_threshold,
        symlog_linthresh=args.symlog_linthresh,
    )
    contour_plot_seconds = time.perf_counter() - plot_start
    plot_start = time.perf_counter()
    leaf_grid = make_tree_leaf_grid(
        tree,
        domain.bounds,
        resolution=args.grid_resolution,
    )
    leaf_grid_seconds = time.perf_counter() - plot_start
    plot_start = time.perf_counter()
    save_tree_leaf_regions(
        tree,
        domain.bounds,
        title="Adaptive regression tree leaf regions",
        out_path=output_dir / "tree_leaf_regions.png",
        resolution=args.grid_resolution,
        label_leaves=args.label_leaves,
        leaf_grid=leaf_grid,
    )
    leaf_plot_seconds = time.perf_counter() - plot_start
    plot_start = time.perf_counter()
    save_tree_leaf_error_regions(
        tree,
        domain.bounds,
        title=f"Leaf training error: {args.error_metric}",
        error_label=args.error_metric,
        out_path=output_dir / "tree_leaf_errors.png",
        resolution=args.grid_resolution,
        label_leaves=args.label_leaves,
        leaf_grid=leaf_grid,
    )
    leaf_error_plot_seconds = time.perf_counter() - plot_start
    leaf_errors = np.asarray([leaf.metadata["fit_error"] for leaf in leaves], dtype=float)

    expected_leaf_statuses = (
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
    report_lines = [
        "[configuration]",
        *(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        f"resolved_output_dir: {output_dir}",
        f"resolved_domain_bounds: {domain.bounds.tolist()}",
        f"dimension: {DIMENSION}",
        f"leaf_model: {type(model).__name__}",
        f"leaf_include_bias: {leaf_include_bias}",
        f"target_samples_per_node: {builder.target_samples}",
        f"resolved_contour_scale: {resolved_contour_scale}",
        "temperature_strategy: tune_root",
        "store_diagnostics: True",
        "store_samples: False",
        "",
        "[target]",
        *(f"{key}: {value}" for key, value in target_metadata.items()),
        "",
        "[performance]",
        f"test_mse: {test_mse:.12e}",
        f"test_rmse: {np.sqrt(test_mse):.12e}",
        f"test_relative_l2_error: {test_relative_l2:.12e}",
        f"test_median_pointwise_relative_error: {test_median_relative:.12e}",
        f"test_max_pointwise_relative_error: {test_max_relative:.12e}",
        f"test_max_absolute_error: {test_max_absolute:.12e}",
        "",
        "[tree]",
        f"oracle_queries: {build_result.oracle_queries}",
        f"restarts_on_failure: {build_result.restarts_on_failure}",
        f"num_nodes: {tree.num_nodes()}",
        f"num_split_nodes: {tree.num_split_nodes()}",
        f"num_leaves: {tree.num_leaves()}",
        f"realized_max_depth: {tree.max_depth()}",
        f"min_leaf_fit_error: {np.min(leaf_errors):.12e}",
        f"max_leaf_fit_error: {np.max(leaf_errors):.12e}",
        f"selected_root_temperature_c: {tree.root.metadata.get('temperature_c')}",
        f"selected_root_temperature: {tree.root.metadata.get('temperature')}",
        "",
        "[timing_seconds]",
        f"tree_build: {build_seconds:.6f}",
        f"test_sampling: {test_sampling_seconds:.6f}",
        f"test_oracle_evaluation: {test_oracle_seconds:.6f}",
        f"tree_prediction: {prediction_seconds:.6f}",
        f"predictions_per_second: {args.n_test / max(prediction_seconds, 1e-12):.3f}",
        f"contour_plot: {contour_plot_seconds:.6f}",
        f"leaf_grid_routing: {leaf_grid_seconds:.6f}",
        f"leaf_region_plot: {leaf_plot_seconds:.6f}",
        f"leaf_error_plot: {leaf_error_plot_seconds:.6f}",
        "",
        "[leaf_status_counts]",
        *(f"{status}: {leaf_statuses.get(status, 0)}" for status in expected_leaf_statuses),
        *(
            f"{status}: {count}"
            for status, count in sorted(leaf_statuses.items())
            if status not in expected_leaf_statuses
        ),
        "",
        "[split_status_counts]",
        *(f"{status}: {count}" for status, count in sorted(split_statuses.items())),
        "",
        "[optimizer_stop_reason_counts]",
        *(f"{reason}: {count}" for reason, count in sorted(optimizer_stop_reasons.items())),
        "",
        "[split_warning_counts]",
        *(f"{warning}: {count}" for warning, count in sorted(split_warnings.items())),
    ]
    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"saved report: {report_path}")

    node_rows = []
    for node in nodes:
        node_rows.append(
            {
                "node_id": node.node_id,
                "leaf_plot_label": leaf_plot_labels.get(id(node)),
                "node_type": "leaf" if isinstance(node, LeafNode) else "split",
                "depth": node.depth,
                "status": node.status,
                "fit_error": node.metadata.get("fit_error"),
                "fit_mse": node.metadata.get("fit_mse"),
                "n_samples": node.metadata.get("n_samples"),
                "n_inherited": node.metadata.get("n_inherited"),
                "n_new": node.metadata.get("n_new"),
                "split_gain_mse": node.metadata.get("split_gain_mse"),
                "relative_split_gain_mse": node.metadata.get("relative_split_gain_mse"),
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
        )
    node_path = output_dir / "node_diagnostics.csv"
    with node_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(node_rows[0]))
        writer.writeheader()
        writer.writerows(node_rows)
    print(f"saved diagnostics: {node_path}")
    print(
        f"test relative L2={test_relative_l2:.4e}, leaves={tree.num_leaves()}, "
        f"oracle queries={build_result.oracle_queries}, build={build_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
