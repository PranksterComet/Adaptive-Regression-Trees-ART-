"""Build and evaluate adaptive regression trees on high-dimensional targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain

from examples.benchmark_functions import (
    GAUSSIAN_DEFAULT_INTERVAL,
    GAUSSIAN_DEFAULT_CONDITION_NUMBER,
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
    fixed_spherical_affine_pieces,
    fixed_spherical_affine_radius,
    default_high_dim_gaussian_mixture,
    sphere_radius_for_box_volume_fraction,
)
from examples.plotting_helpers import save_histogram
from examples.tree_benchmark_helpers import (
    add_output_offset,
    add_tree_training_arguments,
    build_tree_benchmark,
    evaluate_tree_benchmark,
    save_tree_artifact,
    write_benchmark_report,
    write_node_diagnostics,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).with_name("tree_high_dim_benchmark_outputs")
BENCHMARK_DEFAULT_INTERVALS = {
    "quadratic": QUADRATIC_DEFAULT_INTERVAL,
    "gaussian": GAUSSIAN_DEFAULT_INTERVAL,
    "gaussian_mixture": GAUSSIAN_DEFAULT_INTERVAL,
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
    """Generate a reproducible affine or quadratic spherical-benchmark piece."""

    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2.")
    eigenvalues = (
        np.zeros(dimension)
        if degree == 1
        else rng.normal(scale=0.25, size=dimension)
    )
    return QuadraticFunction(
        dimension=dimension,
        beta=float(rng.normal()),
        w=rng.normal(scale=dimension**-0.5, size=dimension),
        Lambda=eigenvalues,
        q_mode="random" if degree == 2 else "identity",
        random_state=rng,
    )


def make_benchmark_target(
    args: argparse.Namespace,
    domain: BoxDomain,
) -> tuple[object, dict[str, object]]:
    """Construct the selected high-dimensional target and report metadata."""

    dimension = domain.dimension
    feature_seed, volume_seed = np.random.SeedSequence(args.seed).spawn(2)
    feature_rng = np.random.default_rng(feature_seed)
    spectrum = np.arange(1, dimension + 1, dtype=float) ** -3

    if args.benchmark == "quadratic":
        target = QuadraticFunction(
            dimension=dimension,
            beta=0.0,
            w=np.zeros(dimension),
            q_mode=args.rotation,
            random_state=feature_rng,
        )
        metadata = {
            "beta": target.beta,
            "rotation_mode": args.rotation,
            "spectrum_diagonal": spectrum.tolist(),
        }
    elif args.benchmark == "gaussian":
        target = GaussianFunction(
            dimension=dimension,
            beta=0.0,
            amplitude=args.gaussian_amplitude,
            sharpness=args.gaussian_sharpness,
            domain_scale=args.gaussian_domain_scale,
            condition_number=args.gaussian_condition_number,
            q_mode=args.rotation,
            random_state=feature_rng,
        )
        metadata = {
            "beta": target.beta,
            "amplitude": target.amplitude,
            "sharpness": target.sharpness,
            "domain_scale": target.domain_scale,
            "condition_number": target.condition_number,
            "rotation_mode": args.rotation,
            "precision_spectrum": target.precision_spectrum.tolist(),
            "mean": target.mean.tolist(),
        }
    elif args.benchmark == "gaussian_mixture":
        target = default_high_dim_gaussian_mixture(
            dimension,
            beta=0.0,
            random_state=feature_rng,
        )
        metadata = {
            "beta": target.beta,
            "weights": target.weights.tolist(),
            "normalize_weights": target.normalize_weights,
            "component_means": [
                component.mean.tolist() for component in target.components
            ],
            "component_amplitudes": [
                component.amplitude for component in target.components
            ],
            "component_sharpness": [
                component.sharpness for component in target.components
            ],
            "component_condition_numbers": [
                component.condition_number for component in target.components
            ],
            "component_precision_spectra": [
                component.precision_spectrum.tolist()
                for component in target.components
            ],
            "rotation_mode": "independent_random",
        }
    elif args.benchmark == "plane_wave":
        random_features = args.plane_wave_normal is None
        target = PlaneWaveFunction(
            dimension=dimension,
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
            random_state=feature_rng,
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
        center = np.mean(domain.bounds, axis=1)
        if args.sphere_radius is not None:
            volume_fraction = None
            radius = float(args.sphere_radius)
            radius_source = "explicit"
            volume_probe_size = None
        elif args.sphere_volume_fraction is not None:
            volume_fraction = args.sphere_volume_fraction
            radius = sphere_radius_for_box_volume_fraction(
                domain.bounds,
                volume_fraction=volume_fraction,
                center=center,
                n_probe=args.sphere_volume_probe_size,
                random_state=np.random.default_rng(volume_seed),
            )
            radius_source = "estimated_volume_fraction"
            volume_probe_size = args.sphere_volume_probe_size
        else:
            volume_fraction = FIXED_SPHERICAL_AFFINE_VOLUME_FRACTION
            radius = fixed_spherical_affine_radius(dimension)
            radius_source = "fixed_50pct_preset"
            volume_probe_size = None
        if args.sphere_piece_mode == "fixed_affine":
            inside, outside = fixed_spherical_affine_pieces(dimension)
        else:
            inside = random_polynomial_piece(
                dimension,
                args.sphere_piece_degree,
                feature_rng,
            )
            outside = random_polynomial_piece(
                dimension,
                args.sphere_piece_degree,
                feature_rng,
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
            "outside_beta": outside.beta,
            "outside_linear_coefficients": outside.w.tolist(),
        }
    elif args.benchmark == "rosenbrock":
        target = RosenbrockFunction(
            dimension=dimension,
            a=args.rosenbrock_a,
            b=args.rosenbrock_b,
        )
        metadata = {"a": target.a, "b": target.b}
    else:
        target = RastriginFunction(dimension=dimension, A=args.rastrigin_a)
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
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_DEFAULT_INTERVALS),
        default="quadratic",
    )
    parser.add_argument("--dimension", "--dim", dest="dimension", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=19)
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help=(
            "Output offset override. Fixed-affine spherical pieces default to "
            f"{FIXED_SPHERICAL_AFFINE_OFFSET}; all other targets default to 0."
        ),
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Independent uniform test points; default is 10000*dimension.",
    )
    parser.add_argument("--test-batch-size", type=int, default=100_000)
    parser.add_argument("--domain-low", type=float, default=None)
    parser.add_argument("--domain-high", type=float, default=None)
    parser.add_argument(
        "--rotation",
        choices=("identity", "random"),
        default="random",
        help="Rotation mode for quadratic and Gaussian spectra.",
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
    parser.add_argument("--plane-wave-frequency", type=float, default=4.0)
    parser.add_argument(
        "--plane-wave-normal",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit plane normal, normalized internally; omit it to draw a "
            "random unit normal and phase."
        ),
    )
    parser.add_argument(
        "--plane-wave-phase",
        type=float,
        default=0.0,
        help="Phase in radians when an explicit plane normal is supplied.",
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
    )
    parser.add_argument(
        "--sphere-piece-mode",
        choices=("random", "fixed_affine"),
        default="fixed_affine",
        help=(
            "Use the fixed affine dimension-2--5 benchmark pieces or generate "
            "random polynomial pieces."
        ),
    )
    parser.add_argument("--rosenbrock-a", type=float, default=1.0)
    parser.add_argument("--rosenbrock-b", type=float, default=100.0)
    parser.add_argument("--rastrigin-a", type=float, default=10.0)
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument(
        "--histogram-log-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    add_tree_training_arguments(parser)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.dimension < 1:
        raise ValueError("--dimension must be at least 1.")
    if args.benchmark == "rosenbrock" and args.dimension < 2:
        raise ValueError("Rosenbrock requires --dimension of at least 2.")
    if args.n_test is not None and args.n_test < 1:
        raise ValueError("--n-test must be positive.")
    if args.test_batch_size < 1:
        raise ValueError("--test-batch-size must be positive.")
    if args.leaf_degree < 1:
        raise ValueError("--leaf-degree must be at least 1.")
    if args.splitter == "hrt" and args.leaf_degree != 1:
        raise ValueError("--splitter hrt requires --leaf-degree 1.")
    if args.relative_error_floor <= 0.0:
        raise ValueError("--relative-error-floor must be positive.")
    if args.histogram_bins < 1:
        raise ValueError("--histogram-bins must be positive.")
    if args.gaussian_sharpness <= 0.0:
        raise ValueError("--gaussian-sharpness must be positive.")
    if args.gaussian_domain_scale <= 0.0:
        raise ValueError("--gaussian-domain-scale must be positive.")
    if args.gaussian_condition_number < 1.0:
        raise ValueError("--gaussian-condition-number must be at least 1.")
    if (
        args.plane_wave_normal is not None
        and len(args.plane_wave_normal) != args.dimension
    ):
        raise ValueError(
            "--plane-wave-normal must contain exactly --dimension values."
        )
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
    ):
        if not 2 <= args.dimension <= 5:
            raise ValueError("fixed affine sphere pieces require 2 <= dimension <= 5.")
        if args.sphere_piece_degree != 1:
            raise ValueError("fixed affine sphere pieces require --sphere-piece-degree 1.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    default_interval = BENCHMARK_DEFAULT_INTERVALS[args.benchmark]
    domain_low = default_interval[0] if args.domain_low is None else args.domain_low
    domain_high = default_interval[1] if args.domain_high is None else args.domain_high
    if domain_low >= domain_high:
        raise ValueError("--domain-low must be less than --domain-high.")
    domain = BoxDomain.hypercube(args.dimension, domain_low, domain_high)
    args.n_test = args.dimension * 10_000 if args.n_test is None else args.n_test

    degree_directory = (
        f"degree_{args.leaf_degree}"
        if args.splitter == "soft_oblique"
        else f"degree_{args.leaf_degree}_hrt"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / args.benchmark
        / degree_directory
        / f"dim_{args.dimension}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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

    run = build_tree_benchmark(args, domain, target)
    tree_path = output_dir / "tree.joblib"
    tree_save_seconds = save_tree_artifact(
        run,
        tree_path,
        args,
        domain,
        target_metadata,
    )
    evaluation = evaluate_tree_benchmark(
        run.tree,
        target,
        domain,
        n_test=args.n_test,
        error_metric=args.error_metric,
        relative_error_floor=args.relative_error_floor,
        random_state=args.test_seed,
        batch_size=args.test_batch_size,
    )

    start = time.perf_counter()
    save_histogram(
        evaluation.selected_point_errors,
        title=f"Uniform test errors: {args.error_metric}",
        xlabel=evaluation.selected_point_error_label,
        out_path=output_dir / "test_error_histogram.png",
        bins=args.histogram_bins,
        log_bins=args.histogram_log_bins,
    )
    histogram_seconds = time.perf_counter() - start
    write_node_diagnostics(run.tree, output_dir / "node_diagnostics.csv")
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
        extra_timings={"test_error_histogram": histogram_seconds},
    )
    print(
        f"test relative L2={evaluation.metrics['test_relative_l2_error']:.4e}, "
        f"leaves={run.tree.num_leaves()}, "
        f"oracle queries={run.build_result.oracle_queries}, "
        f"build={run.build_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
