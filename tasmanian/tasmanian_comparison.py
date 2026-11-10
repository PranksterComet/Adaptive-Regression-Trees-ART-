"""Adaptive local-polynomial interpolation of the shared ART benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
from tasmanian.comparison_helpers import (
    batched,
    error_summary,
    format_comparison_report,
    save_comparison_plot,
)
from oracle_config import add_oracle_arguments, make_oracle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_oracle_arguments(parser)
    parser.add_argument("--order", type=int, default=2, help="Local polynomial order: -1 for maximum available, or a nonnegative integer.")
    parser.add_argument("--initial-depth", type=int, default=2)
    parser.add_argument("--refinement", choices=("fds", "classic"), default="fds")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Surplus threshold; not a pointwise error guarantee.")
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of loaded batches, including the initial grid.")
    parser.add_argument(
        "--n-test",
        type=int,
        help="Uniform test points; default is 10000*dimension.",
    )
    parser.add_argument("--test-seed", type=int, default=19)
    parser.add_argument(
        "--n-gaussian-test",
        type=int,
        help="Active Gaussian weighted test points; default is 10000*dimension.",
    )
    parser.add_argument("--gaussian-test-seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--relative-error-floor", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("max_points", "max_iterations", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_test is not None and args.n_test < 1:
        parser.error("n-test must be positive")
    if args.order < -1 or args.initial_depth < 0:
        parser.error("order must be >= -1 and initial-depth must be >= 0")
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    if not np.isfinite(args.relative_error_floor) or args.relative_error_floor <= 0:
        parser.error("relative-error-floor must be finite and positive")
    if args.test_seed < 0:
        parser.error("test-seed must be nonnegative")
    if args.gaussian_test_seed < 0:
        parser.error("gaussian-test-seed must be nonnegative")
    if args.n_gaussian_test is not None and args.n_gaussian_test < 1:
        parser.error("n-gaussian-test must be positive")
    if args.n_gaussian_test is not None and args.benchmark != "active_subspace_gaussian":
        parser.error("n-gaussian-test applies only to active_subspace_gaussian")
    return args


def format_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    bounds: np.ndarray,
    target_metadata: dict[str, object],
    tasmanian_version: str,
    stop_reason: str,
    training_queries: int,
    n_iterations: int,
    performance: dict[str, float],
    relative_quantiles: dict[float, float],
    truth: np.ndarray,
    timings: dict[str, float],
    gaussian_performance: dict[str, float] | None = None,
    gaussian_relative_quantiles: dict[float, float] | None = None,
    gaussian_truth: np.ndarray | None = None,
    gaussian_timings: dict[str, float] | None = None,
) -> str:
    """Format a sectioned report using the same conventions as ART."""

    return format_comparison_report(
        args=args,
        output_dir=output_dir,
        bounds=bounds,
        target_metadata=target_metadata,
        performance=performance,
        relative_quantiles=relative_quantiles,
        truth=truth,
        method_section="sparse_grid",
        method_lines=(
            f"tasmanian_version: {tasmanian_version}",
            f"stop_reason: {stop_reason}",
            f"training_queries: {training_queries}",
            f"completed_iterations: {n_iterations}",
            f"grid_artifact: {output_dir / 'grid.tsgrid'}",
        ),
        timings=timings,
        gaussian_performance=gaussian_performance,
        gaussian_relative_quantiles=gaussian_relative_quantiles,
        gaussian_truth=gaussian_truth,
        gaussian_timings=gaussian_timings,
        configuration_lines=(
            "grid_type: local_polynomial",
            "grid_rule: localp",
            "output_dimension: 1",
        ),
    )


def save_plot(grid, oracle, bounds, output_dir: Path, batch_size: int, floor: float) -> None:
    save_comparison_plot(
        grid.evaluateBatch,
        "Local polynomial interpolant",
        oracle,
        bounds,
        output_dir,
        batch_size,
        floor,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    oracle, bounds, target_metadata = make_oracle(args)
    args.dimension = target_metadata["dimension"]
    if args.n_test is None:
        args.n_test = args.dimension * 10_000
    if args.benchmark == "active_subspace_gaussian":
        args.n_gaussian_test = (
            args.dimension * 10_000
            if args.n_gaussian_test is None
            else args.n_gaussian_test
        )
    import Tasmanian

    output_name = f"{args.benchmark}_{args.dimension}d"
    if args.benchmark == "active_subspace_gaussian":
        active_dimension = target_metadata["parameters"]["active_dimension"]
        output_name += f"_k{active_dimension}"
    output_name += f"_order{args.order}_{args.refinement}"
    output_dir = args.output_dir or Path(__file__).parent / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Resolved oracle: " + json.dumps(target_metadata, sort_keys=True), flush=True)

    # Cache a fixed independent test set. It never controls refinement or stopping.
    start = perf_counter()
    grid = Tasmanian.makeLocalPolynomialGrid(args.dimension, 1, args.initial_depth, iOrder=args.order, sRule="localp")
    grid.setDomainTransform(bounds)
    training_seconds = perf_counter() - start
    if grid.getNumNeeded() > args.max_points:
        raise ValueError(f"Initial grid needs {grid.getNumNeeded()} points; increase --max-points or reduce --initial-depth")
    start = perf_counter()
    test_points = np.random.default_rng(args.test_seed).uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(args.n_test, args.dimension),
    )
    test_sampling_seconds = perf_counter() - start
    start = perf_counter()
    truth = batched(oracle, test_points, args.batch_size)
    test_oracle_seconds = perf_counter() - start
    gaussian_test_points = None
    gaussian_truth = None
    gaussian_test_sampling_seconds = 0.0
    gaussian_test_oracle_seconds = 0.0
    if args.benchmark == "active_subspace_gaussian":
        start = perf_counter()
        gaussian_test_points = oracle.target.sample_normalized_on_domain(
            args.n_gaussian_test,
            random_state=args.gaussian_test_seed,
        )
        gaussian_test_sampling_seconds = perf_counter() - start
        start = perf_counter()
        gaussian_truth = batched(oracle, gaussian_test_points, args.batch_size)
        gaussian_test_oracle_seconds = perf_counter() - start
    oracle_seconds = 0.0
    history = []
    stop_reason = "max_iterations"
    for iteration in range(args.max_iterations):
        start = perf_counter()
        points = grid.getNeededPoints()
        oracle_start = perf_counter()
        values = batched(oracle, points, args.batch_size)
        oracle_seconds += perf_counter() - oracle_start
        grid.loadNeededPoints(values.reshape(-1, 1))
        training_seconds += perf_counter() - start

        start = perf_counter()
        prediction = batched(grid.evaluateBatch, test_points, args.batch_size)
        prediction_seconds = perf_counter() - start
        performance, relative_quantiles = error_summary(
            truth,
            prediction,
            args.relative_error_floor,
        )
        gaussian_prediction = None
        gaussian_prediction_seconds = 0.0
        gaussian_performance = None
        gaussian_relative_quantiles = None
        if gaussian_test_points is not None and gaussian_truth is not None:
            start = perf_counter()
            gaussian_prediction = batched(
                grid.evaluateBatch,
                gaussian_test_points,
                args.batch_size,
            )
            gaussian_prediction_seconds = perf_counter() - start
            gaussian_performance, gaussian_relative_quantiles = error_summary(
                gaussian_truth,
                gaussian_prediction,
                args.relative_error_floor,
            )
        row = {
            "iteration": iteration,
            "training_queries": grid.getNumLoaded(),
            "training_seconds": training_seconds,
            "training_oracle_seconds": oracle_seconds,
            "test_prediction_seconds": prediction_seconds,
            **performance,
            **{
                f"pointwise_relative_q{100 * quantile:g}": value
                for quantile, value in relative_quantiles.items()
            },
        }
        if gaussian_performance is not None and gaussian_relative_quantiles is not None:
            row.update(
                gaussian_test_prediction_seconds=gaussian_prediction_seconds,
                **{
                    f"gaussian_weighted_{key}": value
                    for key, value in gaussian_performance.items()
                },
                **{
                    f"gaussian_weighted_pointwise_relative_q{100 * quantile:g}": value
                    for quantile, value in gaussian_relative_quantiles.items()
                },
            )
        history.append(row)
        print(
            f"{grid.getNumLoaded():7d} points | "
            f"q99 relative {relative_quantiles[0.99]:.3e} | "
            "below 1%: "
            f"{100 * performance['test_fraction_below_1pct_relative_error']:.2f}% | "
            f"train {training_seconds:.3f}s",
            flush=True,
        )

        if iteration + 1 == args.max_iterations:
            break
        start = perf_counter()
        grid.setSurplusRefinement(args.tolerance, 0, args.refinement)
        training_seconds += perf_counter() - start
        if grid.getNumNeeded() == 0:
            stop_reason = "surplus_tolerance"
            break
        if grid.getNumLoaded() + grid.getNumNeeded() > args.max_points:
            stop_reason = "next_batch_exceeds_point_budget"
            grid.clearRefinement()
            break

    start = perf_counter()
    grid.write(str(output_dir / "grid.tsgrid"))
    grid_save_seconds = perf_counter() - start
    start = perf_counter()
    saved_predictions = {
        "points": test_points,
        "truth": truth,
        "prediction": prediction,
    }
    if gaussian_test_points is not None:
        saved_predictions.update(
            gaussian_points=gaussian_test_points,
            gaussian_truth=gaussian_truth,
            gaussian_prediction=gaussian_prediction,
        )
    np.savez_compressed(output_dir / "test_predictions.npz", **saved_predictions)
    prediction_save_seconds = perf_counter() - start
    start = perf_counter()
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    history_save_seconds = perf_counter() - start
    comparison_plot_seconds = 0.0
    if args.dimension == 2:
        start = perf_counter()
        save_plot(grid, oracle, bounds, output_dir, args.batch_size, args.relative_error_floor)
        comparison_plot_seconds = perf_counter() - start

    training_queries = grid.getNumLoaded()
    timings = {
        "grid_build": training_seconds,
        "training_oracle_evaluation": oracle_seconds,
        "grid_overhead": max(training_seconds - oracle_seconds, 0.0),
        "test_sampling": test_sampling_seconds,
        "test_oracle_evaluation": test_oracle_seconds,
        "grid_prediction": prediction_seconds,
        "predictions_per_second": args.n_test / max(prediction_seconds, 1e-12),
        "mean_training_oracle_query": oracle_seconds / max(training_queries, 1),
        "mean_test_oracle_query": test_oracle_seconds / args.n_test,
        "grid_save": grid_save_seconds,
        "test_predictions_save": prediction_save_seconds,
        "history_save": history_save_seconds,
    }
    if args.dimension == 2:
        timings["comparison_plot"] = comparison_plot_seconds
    gaussian_timings = None
    if gaussian_test_points is not None:
        gaussian_timings = {
            "test_sampling": gaussian_test_sampling_seconds,
            "test_oracle_evaluation": gaussian_test_oracle_seconds,
            "grid_prediction": gaussian_prediction_seconds,
            "predictions_per_second": (
                args.n_gaussian_test / max(gaussian_prediction_seconds, 1e-12)
            ),
        }
    report = format_report(
        args=args,
        output_dir=output_dir,
        bounds=bounds,
        target_metadata=target_metadata,
        tasmanian_version=grid.getVersion(),
        stop_reason=stop_reason,
        training_queries=training_queries,
        n_iterations=len(history),
        performance=performance,
        relative_quantiles=relative_quantiles,
        truth=truth,
        timings=timings,
        gaussian_performance=gaussian_performance,
        gaussian_relative_quantiles=gaussian_relative_quantiles,
        gaussian_truth=gaussian_truth,
        gaussian_timings=gaussian_timings,
    )
    (output_dir / "report.txt").write_text(report)
    print(f"Stopped: {stop_reason}\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
