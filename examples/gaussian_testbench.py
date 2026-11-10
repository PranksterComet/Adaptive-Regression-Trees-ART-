"""Build and evaluate ART on a rotated active-subspace Gaussian."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.tree import RegressionTree, SplitNode

from examples.benchmark_functions import ActiveSubspaceGaussianFunction
from examples.plotting_helpers import save_histogram
from examples.tree_benchmark_helpers import (
    TreeTestEvaluation,
    add_output_offset,
    add_tree_training_arguments,
    build_tree_benchmark,
    evaluate_tree_benchmark,
    save_tree_artifact,
    write_benchmark_report,
    write_node_diagnostics,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).with_name("gaussian_testbench_outputs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(benchmark="active_subspace_gaussian")
    parser.add_argument("--dimension", "--dim", dest="dimension", type=int, default=5)
    parser.add_argument(
        "--active-dimension",
        "--k",
        dest="active_dimension",
        type=int,
        default=3,
    )
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=None,
        help="Active standard deviations; default is 0.2, 0.5, 0.8, ...",
    )
    parser.add_argument(
        "--active-mean",
        type=float,
        nargs="+",
        default=None,
        help="Mean in active coordinates; default is zero.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=19)
    parser.add_argument("--gaussian-test-seed", type=int, default=23)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Uniform test samples; default is 10000*dimension.",
    )
    parser.add_argument(
        "--n-gaussian-test",
        type=int,
        default=None,
        help="Normalized-Gaussian test samples; default is 10000*dimension.",
    )
    parser.add_argument("--test-batch-size", type=int, default=100_000)
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
    if not 1 <= args.active_dimension <= args.dimension:
        raise ValueError("--active-dimension must lie between 1 and dimension.")
    if not np.isfinite(args.a) or args.a <= 0.0:
        raise ValueError("--a must be finite and positive.")
    if not np.isfinite(args.eta) or args.eta < 0.0:
        raise ValueError("--eta must be finite and nonnegative.")
    if args.active_dimension < args.dimension and args.eta == 0.0:
        raise ValueError("--eta must be positive when inactive dimensions exist.")
    for values, name in ((args.tau, "--tau"), (args.active_mean, "--active-mean")):
        if values is not None and len(values) != args.active_dimension:
            raise ValueError(f"{name} requires exactly active-dimension values.")
        if values is not None and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must be finite.")
    if args.tau is not None and np.any(np.asarray(args.tau) <= 0.0):
        raise ValueError("--tau values must be positive.")
    if args.n_test is not None and args.n_test < 1:
        raise ValueError("--n-test must be positive.")
    if args.n_gaussian_test is not None and args.n_gaussian_test < 1:
        raise ValueError("--n-gaussian-test must be positive.")
    if args.test_batch_size < 1:
        raise ValueError("--test-batch-size must be positive.")
    if args.histogram_bins < 1:
        raise ValueError("--histogram-bins must be positive.")
    if args.leaf_degree < 1:
        raise ValueError("--leaf-degree must be at least 1.")
    if args.splitter == "hrt" and args.leaf_degree != 1:
        raise ValueError("--splitter hrt requires --leaf-degree 1.")
    if args.relative_error_floor <= 0.0:
        raise ValueError("--relative-error-floor must be positive.")


def make_target(
    args: argparse.Namespace,
) -> tuple[ActiveSubspaceGaussianFunction, object, dict[str, object]]:
    """Construct the base density, possibly shifted oracle, and metadata."""

    base_target = ActiveSubspaceGaussianFunction(
        dimension=args.dimension,
        active_dimension=args.active_dimension,
        eta=args.eta,
        a=args.a,
        tau=None if args.tau is None else np.asarray(args.tau, dtype=float),
        active_mean=(
            None
            if args.active_mean is None
            else np.asarray(args.active_mean, dtype=float)
        ),
        random_state=args.seed,
    )
    metadata = {
        "benchmark": "active_subspace_gaussian",
        "dimension": base_target.dimension,
        "active_dimension": base_target.active_dimension,
        "inactive_dimension": base_target.dimension - base_target.active_dimension,
        "a": base_target.a,
        "eta": base_target.eta,
        "tau": base_target.tau.tolist(),
        "active_mean": base_target.active_mean.tolist(),
        "active_basis_U": base_target.U.tolist(),
        "inactive_variance_scaled_coordinates": base_target.inactive_variance,
        "rotation_distribution": "Haar",
        "rotation_seed": args.seed,
        "offset": float(args.offset),
        "gaussian_test_distribution": (
            "density proportional to the unshifted target on [-a, a]^d"
        ),
    }
    return base_target, add_output_offset(base_target, args.offset), metadata


def split_alignment_by_depth(
    tree: RegressionTree,
    active_basis: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return squared split-normal projection onto the active subspace by depth."""

    values: dict[int, list[float]] = defaultdict(list)
    for node in tree.iter_nodes():
        if not isinstance(node, SplitNode):
            continue
        norm_squared = float(node.w @ node.w)
        if norm_squared <= 0.0 or not np.isfinite(norm_squared):
            continue
        projected = active_basis.T @ node.w
        alignment = float(projected @ projected) / norm_squared
        values[node.depth].append(float(np.clip(alignment, 0.0, 1.0)))
    return {
        depth: np.asarray(depth_values, dtype=float)
        for depth, depth_values in sorted(values.items())
    }


def alignment_rows(
    alignments: dict[int, np.ndarray],
) -> list[dict[str, float | int]]:
    """Summarize split-normal alignment values at each tree depth."""

    return [
        {
            "depth": depth,
            "count": values.size,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for depth, values in alignments.items()
        if values.size
    ]


def write_alignment_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    fieldnames = ("depth", "count", "mean", "median", "min", "max")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved alignment diagnostics: {path}")


def append_gaussian_report(
    path: Path,
    evaluation: TreeTestEvaluation,
    n_test: int,
    alignment: list[dict[str, float | int]],
    random_alignment: float,
) -> None:
    """Append Gaussian-weighted accuracy and active-subspace diagnostics."""

    total_splits = sum(int(row["count"]) for row in alignment)
    overall_mean = (
        sum(float(row["mean"]) * int(row["count"]) for row in alignment)
        / total_splits
        if total_splits
        else float("nan")
    )
    lines = [
        "",
        "[gaussian_weighted_performance]",
        "distribution: density proportional to unshifted target on [-a, a]^d",
        f"sample_count: {n_test}",
        *(f"{key}: {value:.12e}" for key, value in evaluation.metrics.items()),
        f"test_error_histogram_quantity: {evaluation.selected_point_error_label}",
        "",
        "[gaussian_weighted_oracle_range]",
        f"estimated_min: {evaluation.oracle_min:.12e}",
        f"estimated_max: {evaluation.oracle_max:.12e}",
        f"estimated_span: {evaluation.oracle_max - evaluation.oracle_min:.12e}",
        "",
        "[gaussian_weighted_pointwise_relative_error_quantiles]",
        *(
            f"q{100 * quantile:g}: {value:.12e}"
            for quantile, value in evaluation.pointwise_relative_quantiles.items()
        ),
        "",
        "[gaussian_weighted_timing_seconds]",
        *(f"{key}: {value:.6f}" for key, value in evaluation.timings.items()),
        "",
        "[split_normal_active_subspace_alignment]",
        "definition: ||U^T w||^2 / ||w||^2",
        f"random_direction_expectation: {random_alignment:.12e}",
        f"split_count: {total_splits}",
        f"overall_mean: {overall_mean:.12e}",
        *(
            f"depth_{int(row['depth'])}: count={int(row['count'])}, "
            f"mean={float(row['mean']):.12e}, "
            f"median={float(row['median']):.12e}, "
            f"min={float(row['min']):.12e}, max={float(row['max']):.12e}"
            for row in alignment
        ),
    ]
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    args.n_test = args.dimension * 10_000 if args.n_test is None else args.n_test
    args.n_gaussian_test = (
        args.dimension * 10_000
        if args.n_gaussian_test is None
        else args.n_gaussian_test
    )
    domain = BoxDomain.hypercube(args.dimension, -args.a, args.a)
    base_target, target, target_metadata = make_target(args)

    degree_directory = (
        f"degree_{args.leaf_degree}"
        if args.splitter == "soft_oblique"
        else f"degree_{args.leaf_degree}_hrt"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / degree_directory
        / f"dim_{args.dimension}_active_{args.active_dimension}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run = build_tree_benchmark(args, domain, target)
    tree_path = output_dir / "tree.joblib"
    tree_save_seconds = save_tree_artifact(
        run,
        tree_path,
        args,
        domain,
        target_metadata,
    )
    uniform_evaluation = evaluate_tree_benchmark(
        run.tree,
        target,
        domain,
        n_test=args.n_test,
        error_metric=args.error_metric,
        relative_error_floor=args.relative_error_floor,
        random_state=args.test_seed,
        batch_size=args.test_batch_size,
    )
    gaussian_evaluation = evaluate_tree_benchmark(
        run.tree,
        target,
        domain,
        n_test=args.n_gaussian_test,
        error_metric=args.error_metric,
        relative_error_floor=args.relative_error_floor,
        random_state=args.gaussian_test_seed,
        batch_size=args.test_batch_size,
        point_sampler=base_target.sample_normalized_on_domain,
    )

    start = time.perf_counter()
    save_histogram(
        uniform_evaluation.selected_point_errors,
        title=f"Uniform test errors: {args.error_metric}",
        xlabel=uniform_evaluation.selected_point_error_label,
        out_path=output_dir / "uniform_test_error_histogram.png",
        bins=args.histogram_bins,
        log_bins=args.histogram_log_bins,
    )
    uniform_histogram_seconds = time.perf_counter() - start
    start = time.perf_counter()
    save_histogram(
        gaussian_evaluation.selected_point_errors,
        title=f"Gaussian-weighted test errors: {args.error_metric}",
        xlabel=gaussian_evaluation.selected_point_error_label,
        out_path=output_dir / "gaussian_test_error_histogram.png",
        bins=args.histogram_bins,
        log_bins=args.histogram_log_bins,
    )
    gaussian_histogram_seconds = time.perf_counter() - start

    alignments = split_alignment_by_depth(run.tree, base_target.U)
    rows = alignment_rows(alignments)
    write_alignment_csv(rows, output_dir / "split_alignment_by_depth.csv")
    write_node_diagnostics(run.tree, output_dir / "node_diagnostics.csv")
    report_path = output_dir / "report.txt"
    write_benchmark_report(
        report_path,
        args,
        output_dir,
        domain,
        target_metadata,
        run,
        uniform_evaluation,
        tree_path,
        tree_save_seconds,
        extra_timings={
            "uniform_test_error_histogram": uniform_histogram_seconds,
            "gaussian_test_error_histogram": gaussian_histogram_seconds,
        },
    )
    append_gaussian_report(
        report_path,
        gaussian_evaluation,
        args.n_gaussian_test,
        rows,
        args.active_dimension / args.dimension,
    )
    print(
        f"uniform relative L2={uniform_evaluation.metrics['test_relative_l2_error']:.4e}, "
        "Gaussian-weighted relative L2="
        f"{gaussian_evaluation.metrics['test_relative_l2_error']:.4e}, "
        f"leaves={run.tree.num_leaves()}, "
        f"oracle queries={run.build_result.oracle_queries}, "
        f"build={run.build_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
