"""2D soft oblique splitter test for piecewise polynomial targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.sampling import sample_uniform_box
from art.splitters import SoftObliqueSplitter

from plotting_helpers import plot_boundary
from test_helpers import (
    boundary_misclassification,
    hard_split_predict,
    make_model_template,
    make_piecewise_target,
    polynomial_feature_count,
    random_polynomial_theta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree. Use 1 for affine.")
    parser.add_argument("--sample-multiplier", type=int, default=50, help="n_train per model feature.")
    parser.add_argument("--n-test", type=int, default=10_000, help="Number of test samples.")
    parser.add_argument("--temperature", type=float, default=0.018, help="Soft split temperature.")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Leaf ridge parameter.")
    parser.add_argument("--max-iters", type=int, default=200, help="Maximum optimizer iterations.")
    parser.add_argument("--n-restarts", type=int, default=1, help="Number of random splitter initializations.")
    parser.add_argument(
        "--no-adaptive-alpha",
        dest="adaptive_alpha",
        action="store_false",
        help="Disable adaptive control of the Armijo initial step size.",
    )
    parser.add_argument(
        "--refit-during-line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit weighted models for every Armijo candidate.",
    )
    parser.add_argument("--train-seed", type=int, default=3, help="Training sample seed.")
    parser.add_argument("--test-seed", type=int, default=4, help="Test sample seed.")
    parser.add_argument("--target-seed", type=int, default=17, help="Random polynomial target seed.")
    parser.add_argument("--splitter-seed", type=int, default=11, help="Splitter initialization seed.")
    parser.add_argument("--no-bias", action="store_true", help="Disable polynomial bias term for degree > 1.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.set_defaults(adaptive_alpha=True)
    return parser.parse_args()


def degree_label(degree: int) -> str:
    if degree == 1:
        return "affine"
    if degree == 2:
        return "quadratic"
    return f"degree_{degree}"


def default_output_dir(degree: int) -> Path:
    return Path(__file__).with_name(f"soft_oblique_2D_{degree_label(degree)}_outputs")


def make_target_parameters(
    degree: int,
    d: int,
    include_bias: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    true_w = np.array([1.0, 0.0])
    true_w = true_w / np.linalg.norm(true_w)
    true_z = 0.0

    if degree == 1:
        theta_left = np.array([1.4, -0.6, 0.25])
        theta_right = np.array([-0.7, 1.1, -0.35])
    else:
        theta_left = random_polynomial_theta(d, degree, rng, include_bias=include_bias)
        theta_right = random_polynomial_theta(d, degree, rng, include_bias=include_bias)
        if include_bias:
            theta_left[0] += 0.0
            theta_right[0] -= 0.0
    return theta_left, theta_right, true_w, true_z


def save_history_plot(values: list[float], title: str, ylabel: str, out_path: Path, logy: bool = False) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(values, linewidth=2)
    if logy:
        plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def save_summary(summary: dict[str, object], out_path: Path) -> None:
    lines = [f"{key}: {value}" for key, value in summary.items()]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved summary: {out_path}")


def main() -> None:
    args = parse_args()
    if args.degree < 1:
        raise ValueError("degree must be at least 1.")
    if args.sample_multiplier < 1:
        raise ValueError("sample_multiplier must be positive.")
    if args.n_restarts < 1:
        raise ValueError("n_restarts must be positive.")

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.degree)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    bounds = domain.bounds
    d = domain.dimension
    include_bias = not args.no_bias
    n_features = polynomial_feature_count(d, args.degree, include_bias=include_bias)
    n_train = args.sample_multiplier * n_features
    n_test = args.n_test

    theta_left, theta_right, true_w, true_z = make_target_parameters(
        degree=args.degree,
        d=d,
        include_bias=include_bias,
        rng=np.random.default_rng(args.target_seed),
    )

    scale_factor = 1.0
    theta_left = scale_factor * theta_left
    theta_right = scale_factor * theta_right

    target = make_piecewise_target(
        true_w=true_w,
        true_z=true_z,
        theta_left=theta_left,
        theta_right=theta_right,
        degree=args.degree,
        include_bias=include_bias,
    )

    X_train = sample_uniform_box(bounds, n=n_train, random_state=args.train_seed)
    y_train = target(X_train)

    X_test = sample_uniform_box(bounds, n=n_test, random_state=args.test_seed)
    y_test = target(X_test)

    model_template = make_model_template(args.degree, ridge=args.ridge, include_bias=include_bias)
    parent_model = model_template.clone().fit(X_train, y_train)
    parent_loss = mean_squared_error(y_train, parent_model.predict(X_train))

    splitter = SoftObliqueSplitter(
        model_template=model_template.clone(),
        temperature=args.temperature,
        max_iters=args.max_iters,
        grad_atol=1e-8,
        grad_rtol=1e-5,
        min_side_points=8,
        min_side_fraction=0.05,
        n_restarts=args.n_restarts,
        alpha0=1.0,
        rho=0.5,
        armijo_c=1e-4,
        max_backtracks=25,
        adaptive_alpha=args.adaptive_alpha,
        alpha_min=1e-12,
        alpha_max=1e3,
        alpha_grow=10.0,
        alpha_recovery=10.0,
        heavy_backtrack_threshold=8,
        max_line_search_failures=5,
        weight_floor=1e-12,
        refit_during_line_search=args.refit_during_line_search,
        random_state=args.splitter_seed,
    )

    result = splitter.split(X_train, y_train, parent_model=parent_model, parent_loss=parent_loss)
    y_test_pred = hard_split_predict(X_test, result)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_misclassified, boundary_sign = boundary_misclassification(
        X_train,
        true_w=true_w,
        true_z=true_z,
        learned_w=result.w,
        learned_z=result.z,
    )
    test_misclassified, _ = boundary_misclassification(
        X_test,
        true_w=true_w,
        true_z=true_z,
        learned_w=boundary_sign * result.w,
        learned_z=boundary_sign * result.z,
    )

    metadata = result.metadata
    save_history_plot(
        metadata["soft_loss_history"],
        title="Soft Objective Loss",
        ylabel="soft loss",
        out_path=output_dir / "soft_objective_loss.png",
        logy=True,
    )
    save_history_plot(
        metadata["projected_grad_norm_history"],
        title="Projected Gradient Norm",
        ylabel="||projected grad||",
        out_path=output_dir / "projected_gradient_norm.png",
        logy=True,
    )
    save_history_plot(
        metadata["step_size_history"],
        title="Accepted Step Size",
        ylabel="alpha",
        out_path=output_dir / "step_size_history.png",
        logy=True,
    )
    save_history_plot(
        metadata["backtrack_history"],
        title="Backtracking Steps",
        ylabel="backtracks",
        out_path=output_dir / "backtrack_history.png",
        logy=False,
    )
    plot_boundary(
        X=X_train,
        y=y_train,
        bounds=bounds,
        true_w=true_w,
        true_z=true_z,
        learned_w=result.w,
        learned_z=result.z,
        title=f"Soft oblique splitter boundary ({degree_label(args.degree)})",
        out_path=output_dir / "boundary.png",
        learned_color="green",
        show=False,
    )

    summary = {
        "degree": args.degree,
        "degree_label": degree_label(args.degree),
        "include_bias": include_bias,
        "n_features": n_features,
        "sample_multiplier": args.sample_multiplier,
        "n_train": n_train,
        "n_test": n_test,
        "temperature": splitter.temperature,
        "ridge": args.ridge,
        "max_iters": splitter.max_iters,
        "grad_atol": splitter.grad_atol,
        "grad_rtol": splitter.grad_rtol,
        "min_side_points": splitter.min_side_points,
        "min_side_fraction": splitter.min_side_fraction,
        "n_restarts": splitter.n_restarts,
        "alpha0": splitter.alpha0,
        "rho": splitter.rho,
        "armijo_c": splitter.armijo_c,
        "max_backtracks": splitter.max_backtracks,
        "adaptive_alpha": splitter.adaptive_alpha,
        "alpha_min": splitter.alpha_min,
        "alpha_max": splitter.alpha_max,
        "alpha_grow": splitter.alpha_grow,
        "alpha_recovery": splitter.alpha_recovery,
        "heavy_backtrack_threshold": splitter.heavy_backtrack_threshold,
        "max_line_search_failures": splitter.max_line_search_failures,
        "weight_floor": splitter.weight_floor,
        "refit_during_line_search": splitter.refit_during_line_search,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "target_seed": args.target_seed,
        "splitter_seed": args.splitter_seed,
        "parent_loss": f"{result.parent_loss:.6e}",
        "train_hard_split_loss": f"{result.loss:.6e}",
        "split_gain": f"{result.split_gain:.6e}",
        "test_hard_split_mse": f"{test_mse:.6e}",
        "train_boundary_misclassified": f"{train_misclassified}/{n_train}",
        "test_boundary_misclassified": f"{test_misclassified}/{n_test}",
        "boundary_sign_for_comparison": boundary_sign,
        "n_left": result.n_left,
        "n_right": result.n_right,
        "converged": result.converged,
        "n_iters": result.n_iters,
        "stop_reason": metadata["stop_reason"],
        "true_w": true_w,
        "true_z": true_z,
        "theta_left": theta_left,
        "theta_right": theta_right,
        "learned_w": result.w,
        "learned_z": result.z,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    save_summary(summary, output_dir / "summary.txt")


if __name__ == "__main__":
    main()
