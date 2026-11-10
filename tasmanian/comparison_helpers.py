"""Shared evaluation, reporting, and plotting for surrogate comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from art.metrics import mean_squared_error, pointwise_relative_error, relative_l2_error
from examples.tree_benchmark_helpers import POINTWISE_QUANTILES


def batched(
    function: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluate a scalar-output callable in bounded-memory batches."""

    values = np.empty(len(points))
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        values[start:stop] = np.asarray(function(points[start:stop])).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Oracle or surrogate produced nonfinite values")
    return values


def error_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    floor: float,
) -> tuple[dict[str, float], dict[float, float]]:
    """Compute the same test metrics and relative quantiles as ART reports."""

    residual = np.abs(np.asarray(truth) - np.asarray(prediction))
    relative = pointwise_relative_error(truth, prediction, floor=floor)
    quantiles = dict(
        zip(
            POINTWISE_QUANTILES,
            np.quantile(relative, POINTWISE_QUANTILES),
        )
    )
    mse = mean_squared_error(truth, prediction)
    performance = {
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_relative_l2_error": relative_l2_error(
            truth,
            prediction,
            floor=floor,
        ),
        "test_median_pointwise_relative_error": float(quantiles[0.5]),
        "test_max_pointwise_relative_error": float(quantiles[1.0]),
        "test_max_absolute_error": float(np.max(residual)),
        "test_fraction_below_1pct_relative_error": float(np.mean(relative < 0.01)),
    }
    return performance, {key: float(value) for key, value in quantiles.items()}


def format_comparison_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    bounds: np.ndarray,
    target_metadata: dict[str, object],
    performance: dict[str, float],
    relative_quantiles: dict[float, float],
    truth: np.ndarray,
    method_section: str,
    method_lines: Sequence[str],
    timings: dict[str, float],
    gaussian_performance: dict[str, float] | None = None,
    gaussian_relative_quantiles: dict[float, float] | None = None,
    gaussian_truth: np.ndarray | None = None,
    gaussian_timings: dict[str, float] | None = None,
    configuration_lines: Sequence[str] = (),
) -> str:
    """Format common ART-style sections followed by method diagnostics."""

    parameters = target_metadata["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("Resolved target parameters must be a dictionary.")
    lines = [
        "[configuration]",
        *(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        f"resolved_output_dir: {output_dir}",
        f"resolved_domain_bounds: {bounds.tolist()}",
        f"dimension: {args.dimension}",
        *configuration_lines,
        "",
        "[target]",
        f"benchmark: {target_metadata['benchmark']}",
        *(f"{key}: {value}" for key, value in parameters.items()),
        f"preset_source_report: {target_metadata['preset_source_report']}",
        f"notes: {target_metadata['notes']}",
        f"explicit_overrides: {target_metadata['explicit_overrides']}",
        "",
        "[performance]",
        *(f"{key}: {value:.12e}" for key, value in performance.items()),
        "",
        "[test_oracle_range]",
        f"sample_count: {args.n_test}",
        f"estimated_min: {float(np.min(truth)):.12e}",
        f"estimated_max: {float(np.max(truth)):.12e}",
        f"estimated_span: {float(np.max(truth) - np.min(truth)):.12e}",
        "",
        "[pointwise_relative_error_quantiles]",
        *(
            f"q{100 * quantile:g}: {value:.12e}"
            for quantile, value in relative_quantiles.items()
        ),
    ]
    if gaussian_performance is not None:
        if (
            gaussian_relative_quantiles is None
            or gaussian_truth is None
            or gaussian_timings is None
        ):
            raise ValueError("Gaussian report inputs must be provided together.")
        lines.extend(
            [
                "",
                "[gaussian_weighted_performance]",
                "distribution: density proportional to unshifted target on [-a, a]^d",
                f"sample_count: {args.n_gaussian_test}",
                *(
                    f"{key}: {value:.12e}"
                    for key, value in gaussian_performance.items()
                ),
                "",
                "[gaussian_weighted_oracle_range]",
                f"estimated_min: {float(np.min(gaussian_truth)):.12e}",
                f"estimated_max: {float(np.max(gaussian_truth)):.12e}",
                (
                    "estimated_span: "
                    f"{float(np.max(gaussian_truth) - np.min(gaussian_truth)):.12e}"
                ),
                "",
                "[gaussian_weighted_pointwise_relative_error_quantiles]",
                *(
                    f"q{100 * quantile:g}: {value:.12e}"
                    for quantile, value in gaussian_relative_quantiles.items()
                ),
                "",
                "[gaussian_weighted_timing_seconds]",
                *(f"{key}: {value:.6f}" for key, value in gaussian_timings.items()),
            ]
        )
    lines.extend(
        [
            "",
            f"[{method_section}]",
            *method_lines,
            "",
            "[timing_seconds]",
            *(f"{key}: {value:.6f}" for key, value in timings.items()),
        ]
    )
    return "\n".join(lines) + "\n"


def save_comparison_plot(
    predictor: Callable[[np.ndarray], np.ndarray],
    predictor_title: str,
    oracle: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    output_dir: Path,
    batch_size: int,
    floor: float,
) -> None:
    """Save oracle, surrogate, and pointwise-relative-error panels in 2D."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    xx, yy = np.meshgrid(np.linspace(*bounds[0], 151), np.linspace(*bounds[1], 151))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    truth = batched(oracle, points, batch_size)
    prediction = batched(predictor, points, batch_size)
    relative = np.abs(truth - prediction) / np.maximum(np.abs(truth), floor)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = zip(
        axes,
        (truth, prediction, 100 * relative),
        ("Oracle", predictor_title, "Relative error (%, symlog scale)"),
    )
    for panel, (ax, values, title) in enumerate(panels):
        limits = (
            {"vmin": float(truth.min()), "vmax": float(truth.max())}
            if panel < 2
            else {
                "norm": SymLogNorm(
                    linthresh=1e-3,
                    vmin=0,
                    vmax=max(float(values.max()), 1e-3),
                )
            }
        )
        artist = ax.pcolormesh(
            xx,
            yy,
            values.reshape(xx.shape),
            shading="auto",
            **limits,
        )
        fig.colorbar(artist, ax=ax)
        ax.set(title=title, xlabel="x1", ylabel="x2")
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)
