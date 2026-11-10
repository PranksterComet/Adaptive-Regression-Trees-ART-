"""Build and evaluate an adaptive regression tree for a PyClaw oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain

from examples.plotting_helpers import save_histogram
from examples.tree_benchmark_helpers import (
    add_tree_training_arguments,
    build_tree_benchmark,
    evaluate_tree_benchmark,
    save_tree_artifact,
    write_benchmark_report,
    write_node_diagnostics,
)
from oracles.shallow_water import ShallowWaterSensorOracle


DEFAULT_N_TEST = 10_000
DEFAULT_OUTPUT_ROOT = Path(__file__).with_name(
    "tree_shallow_water_benchmark_outputs"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = ShallowWaterSensorOracle()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=19)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument(
        "--eta-left-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=defaults.eta_left_range,
    )
    parser.add_argument(
        "--eta-right-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=defaults.eta_right_range,
    )
    parser.add_argument(
        "--bump-height-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=defaults.bump_height_range,
    )
    parser.add_argument(
        "--sensor-position",
        type=float,
        default=defaults.sensor_position,
    )
    parser.add_argument("--final-time", type=float, default=defaults.final_time)
    parser.add_argument("--cells", type=int, default=defaults.cells)
    parser.add_argument(
        "--space-time", action="store_true",
        help="Learn h(position, time) with five inputs, including position and time.",
    )
    parser.add_argument("--position-range", type=float, nargs=2, default=(0.0, 5.0))
    parser.add_argument("--time-range", type=float, nargs=2, default=(0.0, 0.75))
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument(
        "--histogram-log-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    add_tree_training_arguments(parser)
    return parser.parse_args(argv)


def make_shallow_water_oracle(
    args: argparse.Namespace,
) -> tuple[ShallowWaterSensorOracle, dict[str, object]]:
    """Construct the configured oracle and its report metadata."""

    oracle = ShallowWaterSensorOracle(
        eta_left_range=tuple(args.eta_left_range),
        eta_right_range=tuple(args.eta_right_range),
        bump_height_range=tuple(args.bump_height_range),
        sensor_position=args.sensor_position,
        final_time=args.final_time,
        cells=args.cells,
        sensor_position_range=tuple(args.position_range) if args.space_time else None,
        time_range=tuple(args.time_range) if args.space_time else None,
    )
    metadata = {
        "input_coordinates": f"normalized [-1, 1]^{oracle.dimension}",
        "parameter_names": oracle.parameter_names,
        "physical_parameter_bounds": oracle.parameter_bounds.tolist(),
        "sensor_position": oracle.sensor_position,
        "final_time": oracle.final_time,
        "cells": oracle.cells,
        "channel_bounds": oracle.channel_bounds,
        "dam_position": oracle.dam_position,
        "bump_center": oracle.bump_center,
        "bump_width": oracle.bump_width,
        "gravity": oracle.gravity,
        "dry_tolerance": oracle.dry_tolerance,
        "quantity_of_interest": "final water depth at sensor",
    }
    if oracle.space_time:
        metadata.pop("sensor_position")
        metadata.pop("final_time")
        metadata.update(
            sensor_position_range=oracle.sensor_position_range,
            time_range=oracle.time_range,
            quantity_of_interest="water depth at input position and time",
            spatial_interpolation="linear interpolation of cell-center depths, including T=0",
        )
    return oracle, metadata


def validate_args(args: argparse.Namespace) -> None:
    if args.n_test < 1:
        raise ValueError("--n-test must be positive.")
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    oracle, target_metadata = make_shallow_water_oracle(args)
    domain = BoxDomain.hypercube(oracle.dimension, -1.0, 1.0)

    degree_directory = (
        f"degree_{args.leaf_degree}"
        if args.splitter == "soft_oblique"
        else f"degree_{args.leaf_degree}_hrt"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / "space_time_5d" / degree_directory
              if args.space_time else DEFAULT_OUTPUT_ROOT / degree_directory)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run = build_tree_benchmark(args, domain, oracle)
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
        oracle,
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
    mean_query_seconds = (
        evaluation.timings["test_oracle_evaluation"] / args.n_test
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
            "mean_test_oracle_query": mean_query_seconds,
            "test_error_histogram": histogram_seconds,
        },
    )
    print(
        f"test relative L2={evaluation.metrics['test_relative_l2_error']:.4e}, "
        f"leaves={run.tree.num_leaves()}, "
        f"oracle queries={run.build_result.oracle_queries}, "
        f"mean test query={mean_query_seconds:.6f}s, "
        f"build={run.build_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
