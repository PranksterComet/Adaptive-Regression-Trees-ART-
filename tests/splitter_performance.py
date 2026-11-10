"""Profile one soft-oblique split across representative regression problems."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.metrics import mean_squared_error
from art.models import (
    PreparedDesign,
    PreparedFeatureModel,
    RegressionModel,
    WeightedRegressionModel,
    model_effective_dimension,
)
from art.presets import DEFAULT_TREE_PRESET
from art.sampling import sample_uniform_box
from art.splitters import SoftObliqueSplitter, SplitNotFoundError
from art.timing import SplitterTimingProfile
from examples.benchmark_functions import RosenbrockFunction
from examples.test_helpers import (
    boundary_misclassification_fraction,
    make_model_template,
    make_piecewise_target,
    random_affine_theta,
    random_polynomial_theta,
    random_unit_vector,
)


GPU_OPTIMIZERS = ("cupy_qr", "cupy_cholesky")


def backend_metadata(backend: str) -> dict[str, Any]:
    """Describe the work performed, independently of the CPU model template."""
    if backend not in GPU_OPTIMIZERS:
        return {"optimizer": backend}
    cholesky = backend == "cupy_cholesky"
    return {
        "optimizer": backend,
        "gpu_solver": "cholesky" if cholesky else "qr",
        "gpu_solver_requested": "normal" if cholesky else "qr",
        "solver_fallback": False,
        "condition_estimation": False,
        "complementary_grams": cholesky,
        "gram_floor_correction": cholesky,
        "rhs_assembly": "direct" if cholesky else "weighted_qr",
    }


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dimension: int
    degree: int
    X: np.ndarray
    y: np.ndarray
    model_template: WeightedRegressionModel
    parent_model: RegressionModel
    parent_loss: float
    prepared_design: PreparedDesign | None
    effective_dimension: int
    true_w: np.ndarray | None = None
    true_z: float | None = None
    boundary_probe: np.ndarray | None = None


def fit_parent(
    model_template: WeightedRegressionModel,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RegressionModel, float, PreparedDesign | None]:
    """Prepare features and fit the untimed parent model once."""

    prepared = (
        model_template.prepare_design(X)
        if isinstance(model_template, PreparedFeatureModel)
        else None
    )
    parent = model_template.clone()
    if prepared is None:
        parent.fit(X, y)
        predictions = parent.predict(X)
    else:
        if not isinstance(parent, PreparedFeatureModel):
            raise TypeError("Prepared parent model must support prepared features.")
        parent.fit_design(prepared, y)
        predictions = parent.predict_design(prepared)
    return parent, mean_squared_error(y, predictions), prepared


def make_case(
    name: str,
    degree: int,
    target_kind: str,
    dimension: int,
    args: argparse.Namespace,
    random_state: int,
) -> BenchmarkCase:
    """Generate one fixed dataset reused across all initialization trials."""

    rng = np.random.default_rng(random_state)
    bounds = np.tile(
        np.array([-args.box_half_width, args.box_half_width]),
        (dimension, 1),
    )
    model_template = make_model_template(
        degree,
        ridge=args.ridge,
        include_bias=True,
        solver=args.ridge_solver,
        auto_rcond_threshold=args.auto_rcond_threshold,
    )
    effective_dimension = model_effective_dimension(model_template, dimension)
    n_samples = args.sample_multiplier * effective_dimension
    X = sample_uniform_box(bounds, n_samples, random_state=rng)

    true_w = None
    true_z = None
    boundary_probe = None
    if target_kind == "piecewise":
        true_w = random_unit_vector(dimension, rng)
        true_z = 0.0
        if degree == 1:
            theta_left = random_affine_theta(dimension, rng)
            theta_right = random_affine_theta(dimension, rng)
        else:
            theta_left = random_polynomial_theta(
                dimension,
                degree,
                rng,
                include_bias=True,
            )
            theta_right = random_polynomial_theta(
                dimension,
                degree,
                rng,
                include_bias=True,
            )
        target = make_piecewise_target(
            true_w,
            true_z,
            theta_left,
            theta_right,
            degree=degree,
            include_bias=True,
        )
        y = target(X)
        boundary_probe = sample_uniform_box(
            bounds,
            args.probe_multiplier * dimension,
            random_state=rng,
        )
    elif target_kind == "rosenbrock":
        y = np.asarray(RosenbrockFunction(dimension)(X), dtype=float)
    else:
        raise ValueError(f"Unknown target kind {target_kind!r}.")

    parent_model, parent_loss, prepared = fit_parent(model_template, X, y)
    return BenchmarkCase(
        name=name,
        dimension=dimension,
        degree=degree,
        X=X,
        y=y,
        model_template=model_template,
        parent_model=parent_model,
        parent_loss=parent_loss,
        prepared_design=prepared,
        effective_dimension=effective_dimension,
        true_w=true_w,
        true_z=true_z,
        boundary_probe=boundary_probe,
    )


def run_trial(
    case: BenchmarkCase,
    args: argparse.Namespace,
    random_state: int,
    optimizer: str = "cpu",
) -> dict[str, Any]:
    """Time exactly one split from one random hyperplane initialization."""

    defaults = DEFAULT_TREE_PRESET.splitter
    timing = SplitterTimingProfile()
    splitter = SoftObliqueSplitter(
        model_template=case.model_template.clone(),
        temperature=args.temperature,
        max_iters=args.max_iters,
        grad_atol=defaults.grad_atol,
        grad_rtol=defaults.grad_rtol,
        min_side_points=getattr(args, "min_side_points", defaults.min_side_points),
        min_side_fraction=getattr(args, "min_side_fraction", defaults.min_side_fraction),
        min_side_fallback=False,
        n_restarts=1,
        alpha0=defaults.alpha0,
        rho=defaults.rho,
        armijo_c=defaults.armijo_c,
        max_backtracks=defaults.max_backtracks,
        adaptive_alpha=defaults.adaptive_alpha,
        alpha_min=defaults.alpha_min,
        alpha_max=defaults.alpha_max,
        alpha_grow=defaults.alpha_grow,
        alpha_recovery=defaults.alpha_recovery,
        heavy_backtrack_threshold=defaults.heavy_backtrack_threshold,
        max_line_search_failures=defaults.max_line_search_failures,
        weight_floor=defaults.weight_floor,
        refit_during_line_search=False,
        random_state=random_state,
        split_timing_profile=timing,
        optimizer=optimizer,
        backtrack_batch_size=getattr(args, "backtrack_batch_size", defaults.backtrack_batch_size),
        Xw_refresh=getattr(args, "Xw_refresh", defaults.Xw_refresh),
        qr_parallel=not getattr(args, "qr_serial", False),
    )

    start = perf_counter()
    try:
        result = splitter.split(
            case.X,
            case.y,
            parent_model=case.parent_model,
            parent_loss=case.parent_loss,
            prepared_design=case.prepared_design,
        )
    except SplitNotFoundError as exc:
        elapsed = perf_counter() - start
        summary = timing.summary(elapsed)
        diagnostics = exc.diagnostics if isinstance(exc.diagnostics, dict) else {}
        return {
            "case": case.name,
            "backend": optimizer,
            **backend_metadata(optimizer),
            "physical_line_search_candidates": diagnostics.get("physical_line_search_candidates", timing.line_search_candidate_count),
            "logical_line_search_candidates": diagnostics.get("logical_line_search_candidates", timing.line_search_candidate_count),
            "speculative_line_search_candidates": diagnostics.get("speculative_line_search_candidates", 0),
            "success": False,
            "failure_reason": exc.reason,
            "stop_reason": diagnostics.get("stop_reason", "split_not_found"),
            "n_iters": diagnostics.get("n_iters", np.nan),
            "total_backtracks": np.nansum(diagnostics.get("backtrack_history", [])),
            "misclassification_fraction": np.nan,
            **summary,
        }

    elapsed = perf_counter() - start
    summary = timing.summary(elapsed)
    misclassification = np.nan
    if case.boundary_probe is not None:
        if case.true_w is None or case.true_z is None:
            raise RuntimeError("Piecewise case is missing its true boundary.")
        misclassification, _ = boundary_misclassification_fraction(
            case.boundary_probe,
            case.true_w,
            case.true_z,
            result.w,
            result.z,
        )
    return {
        "case": case.name,
        "backend": optimizer,
        **backend_metadata(optimizer),
        "physical_line_search_candidates": result.metadata.get("physical_line_search_candidates", timing.line_search_candidate_count),
        "logical_line_search_candidates": result.metadata.get("logical_line_search_candidates", timing.line_search_candidate_count),
        "speculative_line_search_candidates": result.metadata.get("speculative_line_search_candidates", 0),
        "success": True,
        "failure_reason": "",
        "stop_reason": result.stop_reason,
        "n_iters": result.n_iters,
        "total_backtracks": sum(result.metadata.get("backtrack_history", [])),
        "misclassification_fraction": misclassification,
        **summary,
    }


def median(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.nanmedian(values))


def build_report(
    cases: list[BenchmarkCase],
    rows_by_case: dict[tuple[str, int], list[dict[str, Any]]],
    args: argparse.Namespace,
    optimizer: str = "cpu",
) -> str:
    """Format one dimension-scaling grid per benchmark case."""

    defaults = DEFAULT_TREE_PRESET.splitter
    backtrack_batch_size = getattr(args, "backtrack_batch_size", 8)
    device_optimizer = optimizer in GPU_OPTIMIZERS
    details = backend_metadata(optimizer)
    lines = [
        "[timing_categories]",
        "least_squares: all timed left/right model-fit pairs, including soft weighted "
        "fits and the final hard fits; parent fitting and feature preparation are excluded.",
        "backtracking: complete Armijo searches, including candidate projection, "
        "sigmoid/loss evaluation, comparisons, step reduction and loop overhead.",
        "gradient: analytic gradient evaluation followed by tangent projection and "
        "gradient-norm calculation for initial and accepted states.",
        "sigmoid: linear score and sigmoid-gate computation for every initial, "
        "accepted and trial candidate evaluation.",
        "The high-level and operation-level percentages overlap: sigmoid work inside "
        "Armijo is included in both backtracking and sigmoid.",
        "",
        "[configuration]",
        f"dimensions: {args.dimensions}",
        f"box_bounds: [{-args.box_half_width}, {args.box_half_width}]^d",
        f"trials_per_case_and_dimension: {args.trials}",
        "warmup_trials_per_case_and_dimension: 1",
        f"sample_multiplier: {args.sample_multiplier}",
        f"probe_multiplier: {args.probe_multiplier}",
        f"temperature: {args.temperature}",
        f"max_iters: {args.max_iters}",
        f"max_backtracks: {defaults.max_backtracks}",
        "refit_during_line_search: false",
        f"ridge: {args.ridge}",
        f"ridge_solver: {args.ridge_solver}",
        f"auto_rcond_threshold: {args.auto_rcond_threshold}",
        f"seed: {args.seed}",
        "data_generation_timed: false",
        "prepared_design_timed: false",
        "parent_fit_timed: false",
        f"backend: {optimizer}",
        f"optimizer: {details['optimizer']}",
        f"cpu_threads: {args.cpu_threads} (BLAS limit)",
        f"cuda_upload_and_precomputation_timed: {str(optimizer != 'cpu').lower()}",
        f"phase_timing_clock: {'CUDA events; setup uses wall time' if device_optimizer else 'wall time'}",
        f"backtrack_batch_size: {backtrack_batch_size if device_optimizer else 'not applicable'}",
        f"Xw_refresh: {getattr(args, 'Xw_refresh', DEFAULT_TREE_PRESET.splitter.Xw_refresh) if device_optimizer else 'not applicable'}",
        f"qr_parallel: {not getattr(args, 'qr_serial', False) if optimizer == 'cupy_qr' else 'not applicable'}",
    ]
    if device_optimizer:
        lines.extend([
            f"gpu_solver: {details['gpu_solver']}",
            f"gpu_solver_requested: {details['gpu_solver_requested']}",
            "condition_estimation: false",
            "solver_fallback: false",
            f"fit_parallel: {not getattr(args, 'qr_serial', False)}",
            f"GPU timings: LS includes paired {details['gpu_solver']} fits and predictions; gradient includes residuals,",
            "reductions, tangent projection, and Xp_w. Sigmoid trials include frozen loss reductions.",
            "trial_steps counts physical evaluations, including speculative batched candidates.",
            "The CPU controls adaptive steps; total_ms is end-to-end wall time including transfers.",
        ])
        if optimizer == "cupy_qr":
            lines.append("QR is unpivoted, with a diagonal safety check and no condition estimate or fallback.")
        else:
            lines.extend([
                "complementary_grams: true",
                "gram_floor_correction: true",
                "rhs_assembly: direct",
                "Cholesky caches the unweighted Gram, derives the larger-trace side,",
                "and applies independent weight-floor corrections before adding ridge.",
                "Solver status and finite outputs are checked; no condition estimate or fallback.",
            ])

    case_names = list(dict.fromkeys(case.name for case in cases))
    for case_name in case_names:
        matching = {case.dimension: case for case in cases if case.name == case_name}
        lines.extend(
            [
                "",
                f"[case:{case_name}]",
                (
                    "                                           high-level (%)       "
                    "operation-level (%)"
                ),
                (
                    " dim      n      p      success   total_ms      LS_%    "
                    "backtrack_%   grad_%   sigmoid_%   trial_steps   misclass"
                ),
            ]
        )
        for dimension in args.dimensions:
            case = matching[dimension]
            rows = rows_by_case[(case.name, dimension)]
            successes = [row for row in rows if bool(row["success"])]
            misclassification = "-"
            if case.boundary_probe is not None:
                misclassification = (
                    f"{median(successes, 'misclassification_fraction'):.3e}"
                    if successes
                    else "nan"
                )
            lines.append(
                f"{dimension:4d}  {case.X.shape[0]:5d}  "
                f"{case.effective_dimension:5d}  "
                f"{len(successes):3d}/{len(rows):<3d}  "
                f"{1e3 * median(rows, 'total_seconds'):10.3f}  "
                f"{median(rows, 'least_squares_percent'):8.2f}  "
                f"{median(rows, 'backtracking_percent'):11.2f}  "
                f"{median(rows, 'gradient_percent'):7.2f}  "
                f"{median(rows, 'sigmoid_percent'):10.2f}  "
                f"{median(rows, 'line_search_candidate_count'):11.1f}  "
                f"{misclassification:>10s}"
            )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[5, 10, 20, 40],
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sample-multiplier", type=int, default=5)
    parser.add_argument("--probe-multiplier", type=int, default=10_000)
    parser.add_argument("--box-half-width", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--min-side-points", type=int, default=DEFAULT_TREE_PRESET.splitter.min_side_points)
    parser.add_argument("--min-side-fraction", type=float, default=DEFAULT_TREE_PRESET.splitter.min_side_fraction)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--ridge-solver",
        choices=("auto", "normal", "qr", "svd"),
        default="auto",
    )
    parser.add_argument("--auto-rcond-threshold", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cpu-threads", type=int, default=16,
        help="CPU BLAS threads for every backend (default: 16).",
    )
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "cuda", "both", "cuda-qr", "cuda_cholesky", "cuda-cholesky", "all"), default="auto",
        help="auto runs CPU plus each available CUDA backend; uses CPU only without CUDA",
    )
    parser.add_argument(
        "--backtrack-batch-size", type=int, default=DEFAULT_TREE_PRESET.splitter.backtrack_batch_size,
        help="Trial candidates per batch (default: 8).",
    )
    parser.add_argument("--xw-refresh", dest="Xw_refresh", type=int,
                        default=DEFAULT_TREE_PRESET.splitter.Xw_refresh)
    parser.add_argument("--qr-serial", "--solver-serial", dest="qr_serial", action="store_true",
                        default=not DEFAULT_TREE_PRESET.splitter.qr_parallel,
                        help="Run the two GPU QR/Cholesky fits in one stream.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "splitter_performance_outputs",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive.")
    if args.backtrack_batch_size < 1:
        parser.error("--backtrack-batch-size must be positive.")
    if args.Xw_refresh < 0:
        parser.error("--xw-refresh must be nonnegative.")
    if args.min_side_points < 1 or not 0 <= args.min_side_fraction < .5:
        parser.error("Require --min-side-points >= 1 and 0 <= --min-side-fraction < 0.5.")
    if any(dimension < 2 for dimension in args.dimensions):
        parser.error("--dimensions values must be at least 2 for Rosenbrock.")
    args.dimensions = list(dict.fromkeys(args.dimensions))
    if args.trials < 1:
        parser.error("--trials must be positive.")
    if args.sample_multiplier < 1 or args.probe_multiplier < 1:
        parser.error("Sample and probe multipliers must be positive.")
    if args.box_half_width <= 0.0:
        parser.error("--box-half-width must be positive.")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive.")
    if args.max_iters < 0:
        parser.error("--max-iters must be nonnegative.")
    if args.ridge < 0.0:
        parser.error("--ridge must be nonnegative.")
    return args


def select_backends(mode: str, solver: str) -> tuple[list[str], str]:
    """Probe available backends; explicit CUDA requests fail when unavailable."""
    if mode == "cpu":
        return ["cpu"], "CUDA not requested"
    if mode in ("cuda_cholesky", "cuda-cholesky"):
        from art._cuda_cholesky import cuda_cholesky_status

        available, description = cuda_cholesky_status()
        if not available:
            raise RuntimeError(f"CUDA Cholesky requested but unavailable: {description}")
        return ["cupy_cholesky"], description
    if mode == "cuda-qr":
        from art._cuda_qr import cuda_qr_status

        available, qr_status = cuda_qr_status()
        if not available:
            raise RuntimeError(f"CUDA QR requested but unavailable: {qr_status}")
        return ["cupy_qr"], qr_status
    hybrid_supported = solver in ("auto", "normal")
    if mode in ("cuda", "both") and not hybrid_supported:
        raise RuntimeError("CUDA Gram acceleration requires --ridge-solver auto or normal")
    from art._cuda_hybrid_gram import cuda_status

    available, status = cuda_status()
    if not available:
        if mode != "auto":
            raise RuntimeError(f"CUDA requested but unavailable: {status}")
        return ["cpu"], status
    if mode == "cuda":
        return ["hybrid_gram"], status
    if mode == "both":
        return ["cpu", "hybrid_gram"], status

    # auto and all share ordering and probe each resident solver once.
    backends = ["cpu"]
    if hybrid_supported:
        backends.append("hybrid_gram")
    else:
        status += "; hybrid Gram skipped: requires --ridge-solver auto or normal"
    from art._cuda_qr import cuda_qr_status

    qr_available, qr_status = cuda_qr_status()
    if qr_available:
        backends.append("cupy_qr")
        status += f"; QR: {qr_status}"
    elif mode == "all":
        raise RuntimeError(f"CUDA QR requested but unavailable: {qr_status}")
    else:
        status += f"; QR unavailable: {qr_status}"
    from art._cuda_cholesky import cuda_cholesky_status

    cholesky_available, cholesky_status = cuda_cholesky_status()
    if cholesky_available:
        backends.append("cupy_cholesky")
        status += f"; Cholesky: {cholesky_status}"
    elif mode == "all":
        raise RuntimeError(f"CUDA Cholesky requested but unavailable: {cholesky_status}")
    else:
        status += f"; Cholesky unavailable: {cholesky_status}"
    return backends, status


def main() -> None:
    args = parse_args()
    # Load both BLAS libraries before applying limits; SciPy otherwise loads
    # LAPACK lazily during the first fit, after threadpoolctl scans libraries.
    import scipy.linalg
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.cpu_threads, user_api="blas"):
        _run_benchmark(args)


def _run_benchmark(args: argparse.Namespace) -> None:
    backends, cuda_description = select_backends(args.backend, args.ridge_solver)
    print(f"Backends: {', '.join(backends)}; CUDA: {cuda_description}", flush=True)
    specifications = (
        ("piecewise_affine", 1, "piecewise"),
        ("piecewise_quadratic", 2, "piecewise"),
        ("rosenbrock_affine", 1, "rosenbrock"),
        ("rosenbrock_quadratic", 2, "rosenbrock"),
    )
    cases = []
    for case_index, (name, degree, target_kind) in enumerate(specifications):
        for dimension_index, dimension in enumerate(args.dimensions):
            cases.append(
                make_case(
                    name,
                    degree,
                    target_kind,
                    dimension,
                    args,
                    random_state=(
                        args.seed
                        + 100_000 * case_index
                        + 1_000 * dimension_index
                    ),
                )
            )

    rows_by_backend = {backend: {} for backend in backends}
    all_rows_by_backend = {backend: [] for backend in backends}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        for case_index, case in enumerate(cases):
            seed_base = args.seed + 1_000_000 * (case_index + 1)
            print(f"Running {backend}: {case.name}, d={case.dimension}", flush=True)
            run_trial(case, args, random_state=seed_base - 1, optimizer=backend)
            rows = []
            for trial in range(args.trials):
                row = run_trial(case, args, random_state=seed_base + trial, optimizer=backend)
                row["dimension"] = case.dimension
                row["leaf_degree"] = case.degree
                row["n_samples"] = case.X.shape[0]
                row["effective_dimension"] = case.effective_dimension
                row["trial"] = trial
                rows.append(row)
                all_rows_by_backend[backend].append(row)
            rows_by_backend[backend][(case.name, case.dimension)] = rows

        # Preserve completed CPU measurements even if a later CUDA run fails.
        report = build_report(cases, rows_by_backend[backend], args, backend)
        suffix = {"cpu": "", "hybrid_gram": "_cuda", "cupy_qr": "_cuda_qr", "cupy_cholesky": "_cuda_cholesky"}[backend]
        (args.output_dir / f"report{suffix}.txt").write_text(report, encoding="utf-8")
        write_csv(all_rows_by_backend[backend], args.output_dir / f"timings{suffix}.csv")
        print(report, end="")
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    metadata = {
        "backends": backends, "cuda": cuda_description,
        "python": platform.python_version(), "cpu": np.__version__,
        "machine": platform.machine(), "host": platform.node(),
        "cpu_threads": args.cpu_threads,
        "backtrack_batch_size": args.backtrack_batch_size,
        "Xw_refresh": args.Xw_refresh,
        "min_side_points": args.min_side_points,
        "min_side_fraction": args.min_side_fraction,
        "fit_parallel": not args.qr_serial,
        "backend_settings": {backend: backend_metadata(backend) for backend in backends},
        "blas_threadpools": pools,
        "thread_environment": {
            key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
