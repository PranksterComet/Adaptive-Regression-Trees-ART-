"""Example test for the SoftObliqueSplitter on a 2D piecewise affine function."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.models import AffineRidgeModel
from art.sampling import sample_uniform_box
from art.splitters import SoftObliqueSplitter

from plotting_helpers import plot_boundary
from test_helpers import boundary_misclassification, hard_split_predict, piecewise_affine


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
    output_dir = Path(__file__).with_name("soft_oblique_affine_2D_outputs")
    output_dir.mkdir(exist_ok=True)

    domain = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    bounds = domain.bounds
    d = domain.dimension
    n_train = 10 * (d + 1)
    n_test = 10_000

    true_w = np.array([0.8, -2.0])
    true_w = true_w / np.linalg.norm(true_w)
    true_z = 0.0

    theta_left = np.array([1.4, -0.6, 0.25])
    theta_right = np.array([-0.7, 1.1, -0.35])

    X_train = sample_uniform_box(bounds, n=n_train, random_state=3)
    y_train = piecewise_affine(X_train, true_w, true_z, theta_left, theta_right)

    X_test = sample_uniform_box(bounds, n=n_test, random_state=4)
    y_test = piecewise_affine(X_test, true_w, true_z, theta_left, theta_right)

    parent_model = AffineRidgeModel(ridge=1e-8).fit(X_train, y_train)
    parent_loss = mean_squared_error(y_train, parent_model.predict(X_train))

    splitter = SoftObliqueSplitter(
        model_template=AffineRidgeModel(ridge=1e-8),
        temperature=0.018,
        max_iters=200,
        grad_atol=1e-6,
        grad_rtol=1e-5,
        min_side_points=8,
        min_side_fraction=0.05,
        n_restarts=1,
        alpha0=1.0,
        rho=0.5,
        armijo_c=1e-4,
        max_backtracks=25,
        adaptive_alpha=True,
        alpha_min=1e-12,
        alpha_max=1e3,
        alpha_grow=10.0,
        alpha_recovery=10.0,
        heavy_backtrack_threshold=8,
        max_line_search_failures=5,
        weight_floor=1e-12,
        random_state=11,
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
    soft_loss_history = metadata["soft_loss_history"]
    grad_norm_history = metadata["projected_grad_norm_history"]
    step_size_history = metadata["step_size_history"]
    backtrack_history = metadata["backtrack_history"]

    save_history_plot(
        soft_loss_history,
        title="Soft Objective Loss",
        ylabel="soft loss",
        out_path=output_dir / "soft_objective_loss.png",
        logy=True,
    )
    save_history_plot(
        grad_norm_history,
        title="Projected Gradient Norm",
        ylabel="||projected grad||",
        out_path=output_dir / "projected_gradient_norm.png",
        logy=True,
    )
    save_history_plot(
        step_size_history,
        title="Accepted Step Size",
        ylabel="alpha",
        out_path=output_dir / "step_size_history.png",
        logy=True,
    )
    save_history_plot(
        backtrack_history,
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
        title="Soft oblique splitter boundary",
        out_path=output_dir / "boundary.png",
        learned_color="green",
        show=False,
    )

    summary = {
        "n_train": n_train,
        "n_test": n_test,
        "temperature": splitter.temperature,
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
        "random_state": splitter.random_state,
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
        "learned_w": result.w,
        "learned_z": result.z,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    save_summary(summary, output_dir / "summary.txt")


if __name__ == "__main__":
    main()
