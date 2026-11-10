"""Build and evaluate a 2D adaptive regression tree on benchmark targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain

from examples.benchmark_functions import (
    GAUSSIAN_DEFAULT_INTERVAL,
    GAUSSIAN_DEFAULT_CONDITION_NUMBER,
    GAUSSIAN_MIXTURE_DEFAULT_INTERVAL,
    PLANE_WAVE_DEFAULT_INTERVAL,
    QUADRATIC_DEFAULT_INTERVAL,
    RASTRIGIN_DEFAULT_INTERVAL,
    ROSENBROCK_DEFAULT_INTERVAL,
    SPHERICAL_PIECEWISE_DEFAULT_INTERVAL,
    FIXED_SPHERICAL_AFFINE_OFFSET,
    FIXED_SPHERICAL_AFFINE_VOLUME_FRACTION,
    GaussianFunction,
    PlaneWaveFunction,
    QuadraticFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphericalPiecewisePolynomialFunction,
    default_gaussian_mixture_2d,
    fixed_spherical_affine_pieces,
    fixed_spherical_affine_radius,
    random_orthogonal_matrix,
    rotation_matrix_2d,
    sphere_radius_for_box_volume_fraction,
)
from examples.plotting_helpers import (
    make_tree_leaf_grid,
    save_function_contour,
    save_tree_contour,
    save_tree_leaf_error_regions,
    save_tree_leaf_regions,
)
from examples.tree_benchmark_helpers import (
    add_output_offset,
    add_tree_training_arguments,
    build_tree_benchmark,
    evaluate_tree_benchmark,
    save_tree_artifact,
    write_benchmark_report,
    write_node_diagnostics,
)


DIMENSION = 2
DEFAULT_OUTPUT_ROOT = Path(__file__).with_name("tree_2D_benchmark_outputs")
BENCHMARK_DEFAULT_INTERVALS = {
    "quadratic": QUADRATIC_DEFAULT_INTERVAL,
    "gaussian": GAUSSIAN_DEFAULT_INTERVAL,
    "gaussian_mixture": GAUSSIAN_MIXTURE_DEFAULT_INTERVAL,
    "plane_wave": PLANE_WAVE_DEFAULT_INTERVAL,
    "spherical_piecewise": SPHERICAL_PIECEWISE_DEFAULT_INTERVAL,
    "rosenbrock": ROSENBROCK_DEFAULT_INTERVAL,
    "rastrigin": RASTRIGIN_DEFAULT_INTERVAL,
}


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
    domain: BoxDomain,
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
        gaussian_random_rotation = args.gaussian_rotation == "random"
        target = GaussianFunction(
            dimension=DIMENSION,
            beta=0.0,
            amplitude=args.gaussian_amplitude,
            sharpness=args.gaussian_sharpness,
            domain_scale=args.gaussian_domain_scale,
            condition_number=args.gaussian_condition_number,
            Q=None if gaussian_random_rotation else rotation,
            q_mode="random" if gaussian_random_rotation else "identity",
            random_state=args.seed,
        )
        metadata = {
            "beta": target.beta,
            "amplitude": target.amplitude,
            "sharpness": target.sharpness,
            "domain_scale": target.domain_scale,
            "condition_number": target.condition_number,
            "rotation_mode": args.gaussian_rotation,
            "Q": target.Q.tolist(),
            "precision_spectrum": target.precision_spectrum.tolist(),
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
        center = np.mean(domain.bounds, axis=1)
        if args.sphere_radius is not None:
            radius = float(args.sphere_radius)
            radius_source = "explicit"
            volume_fraction = None
            volume_probe_size = None
        elif args.sphere_volume_fraction is not None:
            radius = sphere_radius_for_box_volume_fraction(
                domain.bounds,
                volume_fraction=args.sphere_volume_fraction,
                center=center,
                n_probe=args.sphere_volume_probe_size,
                random_state=args.seed,
            )
            radius_source = "estimated_volume_fraction"
            volume_fraction = args.sphere_volume_fraction
            volume_probe_size = args.sphere_volume_probe_size
        else:
            radius = fixed_spherical_affine_radius(DIMENSION)
            radius_source = "fixed_50pct_preset"
            volume_fraction = FIXED_SPHERICAL_AFFINE_VOLUME_FRACTION
            volume_probe_size = None
        if args.sphere_piece_mode == "fixed_affine":
            inside, outside = fixed_spherical_affine_pieces(DIMENSION)
        else:
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
            radius=radius,
            center=center,
        )
        metadata = {
            "radius": target.radius,
            "radius_source": radius_source,
            "requested_volume_fraction": volume_fraction,
            "volume_probe_size": volume_probe_size,
            "center": target.center.tolist(),
            "piece_degree": args.sphere_piece_degree,
            "piece_mode": args.sphere_piece_mode,
            "inside_beta": inside.beta,
            "inside_linear_coefficients": inside.w.tolist(),
            "inside_quadratic_matrix": inside.quadratic_matrix.tolist(),
            "outside_beta": outside.beta,
            "outside_linear_coefficients": outside.w.tolist(),
            "outside_quadratic_matrix": outside.quadratic_matrix.tolist(),
        }
    elif args.benchmark == "rosenbrock":
        target = RosenbrockFunction(
            dimension=DIMENSION,
            a=args.rosenbrock_a,
            b=args.rosenbrock_b,
        )
        metadata = {"a": target.a, "b": target.b}
    else:
        target = RastriginFunction(
            dimension=DIMENSION,
            A=args.rastrigin_a,
        )
        metadata = {"A": target.A}
    output_offset = 0.0 if args.offset is None else float(args.offset)
    if (
        args.benchmark == "spherical_piecewise"
        and args.sphere_piece_mode == "fixed_affine"
        and args.offset is None
    ):
        output_offset = FIXED_SPHERICAL_AFFINE_OFFSET
    metadata["offset"] = output_offset
    return add_output_offset(target, output_offset), metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Experiment and target settings.
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_DEFAULT_INTERVALS),
        default="quadratic",
    )
    parser.add_argument("--seed", type=int, default=1, help="Tree construction seed.")
    parser.add_argument(
        "--test-seed",
        type=int,
        default=19,
        help="Independent test sample seed.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help=(
            "Output offset override. Fixed-affine spherical pieces default to "
            f"{FIXED_SPHERICAL_AFFINE_OFFSET}; all other targets default to 0."
        ),
    )
    parser.add_argument("--n-test", type=int, default=DIMENSION * 10_000)
    parser.add_argument("--domain-low", type=float, default=None)
    parser.add_argument("--domain-high", type=float, default=None)
    parser.add_argument("--rotation-angle-degrees", type=float, default=15.0)
    parser.add_argument("--spectrum", type=float, nargs=2, default=(1.0, 1.0 / 8.0))
    parser.add_argument("--quadratic-linear", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument(
        "--gaussian-rotation",
        choices=("random", "angle"),
        default="random",
    )
    parser.add_argument("--gaussian-amplitude", type=float, default=1.0)
    parser.add_argument("--gaussian-sharpness", type=float, default=1.0)
    parser.add_argument("--gaussian-domain-scale", type=float, default=1.0)
    parser.add_argument(
        "--gaussian-condition-number",
        type=float,
        default=GAUSSIAN_DEFAULT_CONDITION_NUMBER,
    )
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
    sphere_group = parser.add_mutually_exclusive_group()
    sphere_group.add_argument("--sphere-radius", type=float, default=None)
    sphere_group.add_argument("--sphere-volume-fraction", type=float, default=None)
    parser.add_argument("--sphere-volume-probe-size", type=int, default=100_000)
    parser.add_argument(
        "--sphere-piece-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help="Polynomial degree used when --sphere-piece-mode=random.",
    )
    parser.add_argument(
        "--sphere-piece-mode",
        choices=("random", "fixed_affine"),
        default="fixed_affine",
        help="Use the fixed affine preset or random polynomial pieces.",
    )
    parser.add_argument("--rosenbrock-a", type=float, default=1.0)
    parser.add_argument("--rosenbrock-b", type=float, default=100.0)
    parser.add_argument("--rastrigin-a", type=float, default=10.0)
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
    parser.add_argument("--output-dir", type=Path, default=None)
    add_tree_training_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    default_interval = BENCHMARK_DEFAULT_INTERVALS[args.benchmark]
    domain_low = default_interval[0] if args.domain_low is None else args.domain_low
    domain_high = default_interval[1] if args.domain_high is None else args.domain_high
    if domain_low >= domain_high:
        raise ValueError("--domain-low must be less than --domain-high.")
    if args.sphere_radius is not None and args.sphere_radius <= 0.0:
        raise ValueError("--sphere-radius must be positive.")
    if args.sphere_volume_fraction is not None and not (
        0.0 < args.sphere_volume_fraction < 1.0
    ):
        raise ValueError("--sphere-volume-fraction must lie between 0 and 1.")
    if args.sphere_volume_probe_size < 2:
        raise ValueError("--sphere-volume-probe-size must be at least 2.")
    if (
        args.benchmark == "spherical_piecewise"
        and args.sphere_piece_mode == "fixed_affine"
        and args.sphere_piece_degree != 1
    ):
        raise ValueError("fixed affine sphere pieces require --sphere-piece-degree 1.")
    if args.n_test < 1:
        raise ValueError("--n-test must be positive.")
    if args.leaf_degree < 1:
        raise ValueError("--leaf-degree must be at least 1.")
    if args.splitter == "hrt" and args.leaf_degree != 1:
        raise ValueError("--splitter hrt requires --leaf-degree 1.")
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
    if args.gaussian_sharpness <= 0.0:
        raise ValueError("--gaussian-sharpness must be positive.")
    if args.gaussian_domain_scale <= 0.0:
        raise ValueError("--gaussian-domain-scale must be positive.")
    if args.gaussian_condition_number < 1.0:
        raise ValueError("--gaussian-condition-number must be at least 1.")

    domain = BoxDomain.hypercube(DIMENSION, domain_low, domain_high)
    target, target_metadata = make_benchmark_target(args, domain)
    if args.benchmark == "spherical_piecewise":
        center = np.asarray(target_metadata["center"], dtype=float)
        radius = float(target_metadata["radius"])
        min_distance = float(
            np.linalg.norm(
                center - np.clip(center, domain.bounds[:, 0], domain.bounds[:, 1])
            )
        )
        max_distance = float(
            np.linalg.norm(
                np.maximum(
                    np.abs(domain.bounds[:, 0] - center),
                    np.abs(domain.bounds[:, 1] - center),
                )
            )
        )
        if not min_distance < radius < max_distance:
            raise ValueError(
                "The sphere radius must place part of the domain on each side of the sphere."
            )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / args.benchmark
        / (
            f"degree_{args.leaf_degree}"
            if args.splitter == "soft_oblique"
            else f"degree_{args.leaf_degree}_hrt"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run = build_tree_benchmark(args, domain, target)
    tree = run.tree
    tree_path = output_dir / "tree.joblib"
    tree_save_seconds = save_tree_artifact(
        run,
        tree_path,
        args,
        domain,
        target_metadata,
    )
    evaluation = evaluate_tree_benchmark(
        tree,
        target,
        domain,
        n_test=args.n_test,
        error_metric=args.error_metric,
        relative_error_floor=args.relative_error_floor,
        random_state=args.test_seed,
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
    resolved_tree_contour_scale = save_tree_contour(
        leaf_grid,
        title="Adaptive regression tree prediction",
        out_path=output_dir / "tree_prediction_contour.png",
        levels=args.contour_levels,
        scale=args.contour_scale,
        dynamic_range_threshold=args.contour_dynamic_range_threshold,
        symlog_linthresh=args.symlog_linthresh,
    )
    tree_contour_plot_seconds = time.perf_counter() - plot_start
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

    leaf_plot_labels = {
        id(leaf): f"L{index}" for index, leaf in enumerate(leaf_grid.leaves)
    }
    write_node_diagnostics(
        tree,
        output_dir / "node_diagnostics.csv",
        extra_columns=lambda node: {"leaf_plot_label": leaf_plot_labels.get(id(node))},
    )
    write_benchmark_report(
        output_dir / "report.txt",
        args,
        output_dir,
        domain,
        target_metadata,
        run,
        evaluation,
        tree_path,
        tree_save_seconds,
        extra_timings={
            "contour_plot": contour_plot_seconds,
            "leaf_grid_routing": leaf_grid_seconds,
            "tree_contour_plot": tree_contour_plot_seconds,
            "leaf_region_plot": leaf_plot_seconds,
            "leaf_error_plot": leaf_error_plot_seconds,
        },
        extra_configuration={
            "resolved_contour_scale": resolved_contour_scale,
            "resolved_tree_contour_scale": resolved_tree_contour_scale,
        },
    )
    print(
        "test relative L2="
        f"{evaluation.metrics['test_relative_l2_error']:.4e}, "
        f"leaves={tree.num_leaves()}, "
        f"oracle queries={run.build_result.oracle_queries}, "
        f"build={run.build_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
