"""Random-forest regression on the shared ART benchmark oracles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasmanian.comparison_helpers import (
    batched,
    error_summary,
    format_comparison_report,
    save_comparison_plot,
)
from tasmanian.oracle_config import add_oracle_arguments, make_oracle


def _max_features(value: str) -> int | float | str:
    if value in ("sqrt", "log2"):
        return value
    if value.isdigit() and int(value) >= 1:
        return int(value)
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max-features must be a positive integer, a fraction in (0, 1], "
            "'sqrt', or 'log2'"
        ) from exc
    if 0.0 < numeric <= 1.0:
        return numeric
    raise argparse.ArgumentTypeError(
        "max-features must be a positive integer, a fraction in (0, 1], "
        "'sqrt', or 'log2'"
    )


def _max_samples(value: str) -> int | float:
    if value.isdigit() and int(value) >= 1:
        return int(value)
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max-samples must be a positive integer or a fraction in (0, 1]"
        ) from exc
    if 0.0 < numeric <= 1.0:
        return numeric
    raise argparse.ArgumentTypeError(
        "max-samples must be a positive integer or a fraction in (0, 1]"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_oracle_arguments(parser)
    parser.add_argument(
        "--training-points",
        "--max-points",
        dest="training_points",
        type=int,
        default=10_000,
        help="Uniform oracle queries used to train the forest.",
    )
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--criterion",
        choices=("squared_error", "friedman_mse", "absolute_error", "poisson"),
        default="squared_error",
    )
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=_max_features, default=1.0)
    parser.add_argument("--max-leaf-nodes", type=int)
    parser.add_argument("--min-impurity-decrease", type=float, default=0.0)
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-samples", type=_max_samples)
    parser.add_argument("--ccp-alpha", type=float, default=0.0)
    parser.add_argument(
        "--oob-score",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Use 1 for timing comparisons with the single-threaded ART builder.",
    )
    parser.add_argument("--forest-seed", type=int, default=1)
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

    positive = (
        "training_points",
        "n_estimators",
        "min_samples_split",
        "min_samples_leaf",
        "batch_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_depth", "max_leaf_nodes", "n_test", "n_gaussian_test"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_samples_split < 2:
        parser.error("--min-samples-split must be at least 2")
    if args.min_impurity_decrease < 0.0 or not np.isfinite(args.min_impurity_decrease):
        parser.error("--min-impurity-decrease must be finite and nonnegative")
    if args.ccp_alpha < 0.0 or not np.isfinite(args.ccp_alpha):
        parser.error("--ccp-alpha must be finite and nonnegative")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    if args.max_samples is not None and not args.bootstrap:
        parser.error("--max-samples requires --bootstrap")
    if args.oob_score and not args.bootstrap:
        parser.error("--oob-score requires --bootstrap")
    if not np.isfinite(args.relative_error_floor) or args.relative_error_floor <= 0:
        parser.error("--relative-error-floor must be finite and positive")
    for name in ("training_seed", "forest_seed", "test_seed", "gaussian_test_seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.n_gaussian_test is not None and args.benchmark != "active_subspace_gaussian":
        parser.error("--n-gaussian-test applies only to active_subspace_gaussian")
    return args


def make_forest(args: argparse.Namespace) -> RandomForestRegressor:
    """Construct the fixed paper baseline from resolved CLI arguments."""

    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        min_weight_fraction_leaf=0.0,
        max_features=args.max_features,
        max_leaf_nodes=args.max_leaf_nodes,
        min_impurity_decrease=args.min_impurity_decrease,
        bootstrap=args.bootstrap,
        oob_score=args.oob_score,
        n_jobs=args.n_jobs,
        random_state=args.forest_seed,
        verbose=0,
        warm_start=False,
        ccp_alpha=args.ccp_alpha,
        max_samples=args.max_samples,
        monotonic_cst=None,
    )


def forest_statistics(forest: RandomForestRegressor) -> dict[str, object]:
    """Summarize the fitted ensemble's realized size and depth."""

    depths = np.asarray([tree.tree_.max_depth for tree in forest.estimators_])
    leaves = np.asarray([tree.tree_.n_leaves for tree in forest.estimators_])
    nodes = np.asarray([tree.tree_.node_count for tree in forest.estimators_])
    return {
        "tree_count": len(forest.estimators_),
        "total_nodes": int(np.sum(nodes)),
        "total_leaves": int(np.sum(leaves)),
        "mean_tree_depth": float(np.mean(depths)),
        "max_tree_depth": int(np.max(depths)),
        "mean_leaves_per_tree": float(np.mean(leaves)),
        "max_leaves_per_tree": int(np.max(leaves)),
        "feature_importances": forest.feature_importances_.tolist(),
    }


def format_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    bounds: np.ndarray,
    target_metadata: dict[str, object],
    performance: dict[str, float],
    relative_quantiles: dict[float, float],
    truth: np.ndarray,
    timings: dict[str, float],
    statistics: dict[str, object],
    model_bytes: int,
    gaussian_performance: dict[str, float] | None = None,
    gaussian_relative_quantiles: dict[float, float] | None = None,
    gaussian_truth: np.ndarray | None = None,
    gaussian_timings: dict[str, float] | None = None,
) -> str:
    method_lines = [
        f"sklearn_version: {sklearn.__version__}",
        "fit_status: completed",
        "training_distribution: independent uniform points over the domain",
        f"training_queries: {args.training_points}",
        *(f"{key}: {value}" for key, value in statistics.items()),
        f"serialized_model_bytes: {model_bytes}",
        f"model_artifact: {output_dir / 'forest.joblib'}",
    ]
    return format_comparison_report(
        args=args,
        output_dir=output_dir,
        bounds=bounds,
        target_metadata=target_metadata,
        performance=performance,
        relative_quantiles=relative_quantiles,
        truth=truth,
        method_section="random_forest",
        method_lines=method_lines,
        timings=timings,
        gaussian_performance=gaussian_performance,
        gaussian_relative_quantiles=gaussian_relative_quantiles,
        gaussian_truth=gaussian_truth,
        gaussian_timings=gaussian_timings,
        configuration_lines=(
            "surrogate_type: random_forest_regressor",
            "output_dimension: 1",
        ),
    )


def _history_row(
    args: argparse.Namespace,
    timings: dict[str, float],
    performance: dict[str, float],
    relative_quantiles: dict[float, float],
    gaussian_prediction_seconds: float,
    gaussian_performance: dict[str, float] | None,
    gaussian_relative_quantiles: dict[float, float] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "iteration": 0,
        "training_queries": args.training_points,
        "training_sampling_seconds": timings["training_sampling"],
        "training_oracle_seconds": timings["training_oracle_evaluation"],
        "forest_fit_seconds": timings["forest_fit"],
        "test_prediction_seconds": timings["forest_prediction"],
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
    return row


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

    output_name = f"{args.benchmark}_{args.dimension}d"
    if args.benchmark == "active_subspace_gaussian":
        active_dimension = target_metadata["parameters"]["active_dimension"]
        output_name += f"_k{active_dimension}"
    output_name += f"_n{args.training_points}_trees{args.n_estimators}"
    output_dir = args.output_dir or Path(__file__).parent / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Resolved oracle: " + json.dumps(target_metadata, sort_keys=True), flush=True)

    start = perf_counter()
    training_points = np.random.default_rng(args.training_seed).uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(args.training_points, args.dimension),
    )
    training_sampling_seconds = perf_counter() - start
    start = perf_counter()
    training_values = batched(oracle, training_points, args.batch_size)
    training_oracle_seconds = perf_counter() - start

    forest = make_forest(args)
    start = perf_counter()
    forest.fit(training_points, training_values)
    forest_fit_seconds = perf_counter() - start

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
    start = perf_counter()
    prediction = batched(forest.predict, test_points, args.batch_size)
    prediction_seconds = perf_counter() - start
    performance, relative_quantiles = error_summary(
        truth,
        prediction,
        args.relative_error_floor,
    )

    gaussian_test_points = None
    gaussian_truth = None
    gaussian_prediction = None
    gaussian_performance = None
    gaussian_relative_quantiles = None
    gaussian_test_sampling_seconds = 0.0
    gaussian_test_oracle_seconds = 0.0
    gaussian_prediction_seconds = 0.0
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
        start = perf_counter()
        gaussian_prediction = batched(
            forest.predict,
            gaussian_test_points,
            args.batch_size,
        )
        gaussian_prediction_seconds = perf_counter() - start
        gaussian_performance, gaussian_relative_quantiles = error_summary(
            gaussian_truth,
            gaussian_prediction,
            args.relative_error_floor,
        )

    model_path = output_dir / "forest.joblib"
    start = perf_counter()
    joblib.dump(forest, model_path, compress=3)
    model_save_seconds = perf_counter() - start
    model_bytes = model_path.stat().st_size

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

    timings = {
        "total_training": (
            training_sampling_seconds
            + training_oracle_seconds
            + forest_fit_seconds
        ),
        "training_sampling": training_sampling_seconds,
        "training_oracle_evaluation": training_oracle_seconds,
        "forest_fit": forest_fit_seconds,
        "test_sampling": test_sampling_seconds,
        "test_oracle_evaluation": test_oracle_seconds,
        "forest_prediction": prediction_seconds,
        "predictions_per_second": args.n_test / max(prediction_seconds, 1e-12),
        "mean_training_oracle_query": (
            training_oracle_seconds / args.training_points
        ),
        "mean_test_oracle_query": test_oracle_seconds / args.n_test,
        "model_save": model_save_seconds,
        "test_predictions_save": prediction_save_seconds,
    }
    gaussian_timings = None
    if gaussian_test_points is not None:
        gaussian_timings = {
            "test_sampling": gaussian_test_sampling_seconds,
            "test_oracle_evaluation": gaussian_test_oracle_seconds,
            "forest_prediction": gaussian_prediction_seconds,
            "predictions_per_second": (
                args.n_gaussian_test / max(gaussian_prediction_seconds, 1e-12)
            ),
        }

    history = _history_row(
        args,
        timings,
        performance,
        relative_quantiles,
        gaussian_prediction_seconds,
        gaussian_performance,
        gaussian_relative_quantiles,
    )
    start = perf_counter()
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history))
        writer.writeheader()
        writer.writerow(history)
    timings["history_save"] = perf_counter() - start

    if args.dimension == 2:
        start = perf_counter()
        save_comparison_plot(
            forest.predict,
            "Random forest",
            oracle,
            bounds,
            output_dir,
            args.batch_size,
            args.relative_error_floor,
        )
        timings["comparison_plot"] = perf_counter() - start

    statistics = forest_statistics(forest)
    report = format_report(
        args=args,
        output_dir=output_dir,
        bounds=bounds,
        target_metadata=target_metadata,
        performance=performance,
        relative_quantiles=relative_quantiles,
        truth=truth,
        timings=timings,
        statistics=statistics,
        model_bytes=model_bytes,
        gaussian_performance=gaussian_performance,
        gaussian_relative_quantiles=gaussian_relative_quantiles,
        gaussian_truth=gaussian_truth,
        gaussian_timings=gaussian_timings,
    )
    (output_dir / "report.txt").write_text(report)
    print(
        f"{args.training_points} points | "
        f"q99 relative {relative_quantiles[0.99]:.3e} | "
        f"below 1%: {100 * performance['test_fraction_below_1pct_relative_error']:.2f}% | "
        f"fit {forest_fit_seconds:.3f}s",
        flush=True,
    )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
