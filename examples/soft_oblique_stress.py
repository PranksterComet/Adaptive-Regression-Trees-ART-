"""High-dimensional stress test for polynomial SoftObliqueSplitter targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.models import PreparedDesign, PreparedFeatureModel, WeightedRegressionModel
from art.sampling import sample_uniform_box
from art.splitters import SoftObliqueSplitter
from art.temperature import estimate_temperature

from plotting_helpers import save_bar_counts, save_histogram
from test_helpers import (
    boundary_errors,
    boundary_misclassification_fraction,
    hard_split_predict,
    make_model_template,
    make_piecewise_target,
    parse_csv_floats,
    parse_csv_strings,
    polynomial_feature_count,
    random_affine_theta,
    random_polynomial_theta,
    sample_balanced_boundary,
)


def run_candidate(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    temperature: float,
    model_template: WeightedRegressionModel,
    parent_model: WeightedRegressionModel,
    parent_loss: float,
    fit_design: PreparedDesign | None,
    validation_design: PreparedDesign | None,
    args: argparse.Namespace,
    seed: int,
):
    splitter = SoftObliqueSplitter(
        model_template=model_template.clone(),
        temperature=temperature,
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
        adaptive_alpha=not args.disable_adaptive_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_grow=args.alpha_grow,
        alpha_recovery=args.alpha_recovery,
        heavy_backtrack_threshold=args.heavy_backtrack_threshold,
        max_line_search_failures=args.max_line_search_failures,
        weight_floor=args.weight_floor,
        refit_during_line_search=args.refit_during_line_search,
        random_state=seed,
    )
    result = splitter.split(
        X_fit,
        y_fit,
        parent_model=parent_model,
        parent_loss=parent_loss,
        prepared_design=fit_design,
    )
    validation_predictions = (
        hard_split_predict(X_val, result)
        if validation_design is None
        else result.predict_prepared(X_val, validation_design)
    )
    val_mse = mean_squared_error(y_val, validation_predictions)
    return result, val_mse


def degree_label(degree: int) -> str:
    if degree == 1:
        return "affine"
    if degree == 2:
        return "quadratic"
    return f"degree_{degree}"


def model_include_bias(degree: int, no_bias: bool) -> bool:
    return True if degree == 1 else not no_bias


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved csv: {out_path}")


def write_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    successes = [row for row in rows if row["success"]]
    failures = len(rows) - len(successes)
    lines = [
        f"n_trials: {len(rows)}",
        f"successes: {len(successes)}",
        f"failures: {failures}",
    ]
    if successes:
        for key in ("test_mse", "test_misclassification_fraction", "angle_error_degrees", "offset_error"):
            values = np.array([float(row[key]) for row in successes], dtype=float)
            lines.append(f"{key}_mean: {np.mean(values):.8e}")
            lines.append(f"{key}_median: {np.median(values):.8e}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved summary: {out_path}")


def save_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    successes = [row for row in rows if row["success"]]
    if not successes:
        return

    save_histogram(
        [float(row["test_mse"]) for row in successes],
        title="Test MSE",
        xlabel="MSE",
        out_path=output_dir / "hist_test_mse.png",
        bins=30,
        logy=False,
    )
    save_histogram(
        [float(row["test_misclassification_fraction"]) for row in successes],
        title="Boundary Misclassification Fraction",
        xlabel="fraction",
        out_path=output_dir / "hist_misclassification_fraction.png",
        bins=30,
        logy=False,
    )
    save_histogram(
        [float(row["selected_temperature"]) for row in successes],
        title="Selected Temperature",
        xlabel="temperature",
        out_path=output_dir / "hist_selected_temperature.png",
        bins=30,
        logy=False,
    )
    save_histogram(
        [float(row["angle_error_degrees"]) for row in successes],
        title="Boundary Angle Error",
        xlabel="degrees",
        out_path=output_dir / "hist_angle_error.png",
        bins=30,
        logy=False,
    )
    save_histogram(
        [float(row["n_iters"]) for row in successes],
        title="Optimizer Iterations",
        xlabel="iterations",
        out_path=output_dir / "hist_n_iters.png",
        bins=30,
        logy=False,
    )
    save_histogram(
        [float(row["initial_grad_norm"]) for row in successes],
        title="Initial Projected Gradient Norm",
        xlabel="||projected grad||",
        out_path=output_dir / "hist_initial_grad_norm.png",
        bins=30,
        logy=True,
        log_bins=True,
    )
    save_histogram(
        [float(row["grad_norm_ratio"]) for row in successes],
        title="Final / Initial Projected Gradient Norm",
        xlabel="final grad norm / initial grad norm",
        out_path=output_dir / "hist_grad_norm_ratio.png",
        bins=30,
        logy=False,
        log_bins=True,
    )
    save_bar_counts(
        [str(row["selected_c"]) for row in successes],
        title="Selected c Value",
        xlabel="c",
        out_path=output_dir / "bar_selected_c.png",
    )
    save_bar_counts(
        [str(row["selected_mode"]) for row in successes],
        title="Selected Temperature Mode",
        xlabel="mode",
        out_path=output_dir / "bar_selected_mode.png",
    )
    save_bar_counts(
        [str(row["stop_reason"]) for row in successes],
        title="Stop Reason",
        xlabel="stop reason",
        out_path=output_dir / "bar_stop_reason.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--no-bias", action="store_true", help="Disable polynomial bias term for degree > 1.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low", type=float, default=-1.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--sample-multiplier", type=int, default=50)
    parser.add_argument("--c-values", type=str, default="1e-4,0.005,0.01,0.05,0.1,0.5,1.0")
    parser.add_argument("--temperature-modes", type=str, default="median_nn,median_pairwise_scaled")
    parser.add_argument("--max-temp-points", type=int, default=512)
    parser.add_argument("--use-all-temp-points", action="store_true")
    parser.add_argument("--bruteforce-dim-threshold", type=int, default=20)
    parser.add_argument("--min-volume-fraction", type=float, default=0.1)
    parser.add_argument("--boundary-max-attempts", type=int, default=10)
    parser.add_argument("--probe-size", type=int, default=20_000)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--grad-atol", type=float, default=1e-8)
    parser.add_argument("--grad-rtol", type=float, default=1e-5)
    parser.add_argument("--min-side-points", type=int, default=8)
    parser.add_argument("--min-side-fraction", type=float, default=0.05)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--armijo-c", type=float, default=1e-4)
    parser.add_argument("--max-backtracks", type=int, default=25)
    parser.add_argument("--disable-adaptive-alpha", action="store_true")
    parser.add_argument("--alpha-min", type=float, default=1e-12)
    parser.add_argument("--alpha-max", type=float, default=1e3)
    parser.add_argument("--alpha-grow", type=float, default=10.0)
    parser.add_argument("--alpha-recovery", type=float, default=10.0)
    parser.add_argument("--heavy-backtrack-threshold", type=int, default=8)
    parser.add_argument("--max-line-search-failures", type=int, default=5)
    parser.add_argument("--weight-floor", type=float, default=1e-12)
    parser.add_argument(
        "--refit-during-line-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refit weighted models for every Armijo candidate instead of freezing them.",
    )
    args = parser.parse_args()

    if args.dim < 1:
        raise ValueError("--dim must be at least 1.")
    if args.degree < 1:
        raise ValueError("--degree must be at least 1.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1.")
    if args.low >= args.high:
        raise ValueError("--low must be less than --high.")
    if args.sample_multiplier < 1:
        raise ValueError("--sample-multiplier must be at least 1.")

    rng = np.random.default_rng(args.seed)
    d = args.dim
    include_bias = model_include_bias(args.degree, args.no_bias)
    n_features = polynomial_feature_count(d, args.degree, include_bias=include_bias)
    n_fit = args.sample_multiplier * n_features
    n_val = n_fit
    n_test = int(d * 10_000)
    domain = BoxDomain.hypercube(d, args.low, args.high)
    c_values = parse_csv_floats(args.c_values)
    temperature_modes = parse_csv_strings(args.temperature_modes)
    max_temp_points = None if args.use_all_temp_points else args.max_temp_points
    nn_method = "bruteforce" if d >= args.bruteforce_dim_threshold else "kdtree"

    output_dir = Path(__file__).with_name("soft_oblique_stress_outputs") / degree_label(args.degree) / f"dim_{d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for trial in range(args.n_trials):
        start = time.perf_counter()
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        base_row = {
            "trial": trial,
            "success": False,
            "dimension": d,
            "degree": args.degree,
            "degree_label": degree_label(args.degree),
            "include_bias": include_bias,
            "n_features": n_features,
            "n_fit": n_fit,
            "n_val": n_val,
            "n_test": n_test,
            "sample_multiplier": args.sample_multiplier,
            "trial_seed": trial_seed,
            "grad_atol": args.grad_atol,
            "grad_rtol": args.grad_rtol,
            "refit_during_line_search": args.refit_during_line_search,
        }

        try:
            true_w, true_z, true_frac_right = sample_balanced_boundary(
                domain.bounds,
                trial_rng,
                min_volume_fraction=args.min_volume_fraction,
                max_attempts=args.boundary_max_attempts,
                probe_size=args.probe_size,
            )
            if args.degree == 1:
                theta_left = random_affine_theta(d, trial_rng)
                theta_right = random_affine_theta(d, trial_rng)
            else:
                theta_left = random_polynomial_theta(d, args.degree, trial_rng, include_bias=include_bias)
                theta_right = random_polynomial_theta(d, args.degree, trial_rng, include_bias=include_bias)
            target = make_piecewise_target(
                true_w=true_w,
                true_z=true_z,
                theta_left=theta_left,
                theta_right=theta_right,
                degree=args.degree,
                include_bias=include_bias,
            )

            X_fit = sample_uniform_box(domain.bounds, n_fit, random_state=trial_rng)
            y_fit = target(X_fit)
            X_val = sample_uniform_box(domain.bounds, n_val, random_state=trial_rng)
            y_val = target(X_val)
            X_test = sample_uniform_box(domain.bounds, n_test, random_state=trial_rng)
            y_test = target(X_test)
            model_template = make_model_template(
                args.degree,
                ridge=args.ridge,
                include_bias=include_bias,
            )
            fit_design = (
                model_template.prepare_design(X_fit)
                if isinstance(model_template, PreparedFeatureModel)
                else None
            )
            validation_design = (
                model_template.prepare_design(X_val)
                if isinstance(model_template, PreparedFeatureModel)
                else None
            )
            parent_model = model_template.clone()
            if fit_design is None:
                parent_model.fit(X_fit, y_fit)
                parent_predictions = parent_model.predict(X_fit)
            else:
                parent_model.fit_design(fit_design, y_fit)
                parent_predictions = parent_model.predict_design(fit_design)
            parent_loss = mean_squared_error(y_fit, parent_predictions)

            best = None
            candidates = []
            for mode in temperature_modes:
                for c in c_values:
                    temperature = estimate_temperature(
                        X_fit,
                        mode=mode,
                        c=c,
                        max_points=max_temp_points,
                        random_state=trial_seed,
                        nn_method=nn_method,
                    )
                    try:
                        result, val_mse = run_candidate(
                            X_fit,
                            y_fit,
                            X_val,
                            y_val,
                            temperature=temperature,
                            model_template=model_template,
                            parent_model=parent_model,
                            parent_loss=parent_loss,
                            fit_design=fit_design,
                            validation_design=validation_design,
                            args=args,
                            seed=trial_seed,
                        )
                    except Exception as exc:
                        candidates.append((np.inf, mode, c, temperature, None, repr(exc)))
                        continue
                    candidates.append((val_mse, mode, c, temperature, result, ""))
                    if best is None or val_mse < best[0]:
                        best = (val_mse, mode, c, temperature, result)

            if best is None:
                raise RuntimeError("All temperature candidates failed.")

            val_mse, selected_mode, selected_c, selected_temperature, result = best
            y_test_pred = hard_split_predict(X_test, result)
            test_mse = mean_squared_error(y_test, y_test_pred)
            grad_history = result.metadata["projected_grad_norm_history"]
            initial_grad_norm = float(grad_history[0]) if grad_history else np.nan
            final_grad_norm = float(result.metadata["final_grad_norm"])
            grad_norm_ratio = final_grad_norm / initial_grad_norm if initial_grad_norm > 0.0 else np.nan
            test_misfrac, boundary_sign = boundary_misclassification_fraction(
                X_test,
                true_w,
                true_z,
                result.w,
                result.z,
            )
            angle_error, offset_error = boundary_errors(true_w, true_z, result.w, result.z, boundary_sign)
            elapsed = time.perf_counter() - start

            row = {
                **base_row,
                "success": True,
                "selected_mode": selected_mode,
                "selected_c": selected_c,
                "selected_temperature": selected_temperature,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "test_misclassification_fraction": test_misfrac,
                "angle_error_degrees": angle_error,
                "offset_error": offset_error,
                "true_z": true_z,
                "true_frac_right": true_frac_right,
                "learned_z": result.z,
                "n_left": result.n_left,
                "n_right": result.n_right,
                "split_gain": result.split_gain,
                "n_iters": result.n_iters,
                "initial_grad_norm": initial_grad_norm,
                "final_grad_norm": final_grad_norm,
                "grad_norm_ratio": grad_norm_ratio,
                "stop_reason": result.metadata["stop_reason"],
                "elapsed_seconds": elapsed,
            }
        except Exception as exc:
            row = {
                **base_row,
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - start,
            }

        rows.append(row)
        print(f"trial {trial + 1}/{args.n_trials}: success={row['success']}")

    write_csv(rows, output_dir / "summary.csv")
    write_summary(rows, output_dir / "summary.txt")
    save_plots(rows, output_dir)


if __name__ == "__main__":
    main()
