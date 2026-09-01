"""Example test for the HingeAffineSplitter on 2D hinge functions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.models import AffineRidgeModel, augment_features
from art.sampling import sample_uniform_box
from art.splitters import HingeAffineSplitter

from plotting_helpers import boundary_from_thetas, plot_boundary


def affine_values(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return augment_features(X) @ theta


def run_case(mode: str, bounds: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> None:
    X = sample_uniform_box(bounds, n=300, random_state=7)
    y1 = affine_values(X, theta1)
    y2 = affine_values(X, theta2)
    y = np.maximum(y1, y2) if mode == "max" else np.minimum(y1, y2)

    parent_model = AffineRidgeModel(ridge=1e-8).fit(X, y)

    splitter = HingeAffineSplitter(
        mode=mode,
        ridge=1e-8,
        solver="auto",
        auto_rcond_threshold=1e-10,
        mu=1.0,
        max_iters=100,
        tol=1e-8,
        min_side_points=8,
        min_side_fraction=0.05,
        n_restarts=20,
        init_scale=1e-2,
        random_state=11,
    )

    result = splitter.split(X, y, parent_model=parent_model)
    true_w, true_z = boundary_from_thetas(theta1, theta2)

    print(f"\n{mode.upper()} hinge")
    print(f"mode selected: {result.metadata['mode']}")
    print(f"parent_loss: {result.parent_loss:.6e}")
    print(f"split_loss: {result.loss:.6e}")
    print(f"split_gain: {result.split_gain:.6e}")
    print(f"n_left/n_right: {result.n_left}/{result.n_right}")
    print(f"converged: {result.converged}")
    print(f"n_iters: {result.n_iters}")
    print(f"true_w: {true_w}, true_z: {true_z:.6e}")
    print(f"learned_w: {result.w}, learned_z: {result.z:.6e}")

    plot_boundary(
        X=X,
        y=y,
        bounds=bounds,
        true_w=true_w,
        true_z=true_z,
        learned_w=result.w,
        learned_z=result.z,
        title=f"HRT {mode} hinge boundary",
        out_path=Path(__file__).with_name(f"HRT_{mode}_boundary.png"),
    )


def main() -> None:
    domain = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    bounds = domain.bounds

    # Boundary: (theta1 - theta2)^T [x, 1] = 0 -> x1 - x2 + 0.1 = 0.
    # The line crosses the box, so the hinge boundary lies inside the domain.
    theta1 = np.array([1.2, -0.4, 0.3])
    theta2 = np.array([0.2, 0.6, 0.2])

    run_case("max", bounds, theta1, theta2)
    run_case("min", bounds, theta1, theta2)


if __name__ == "__main__":
    main()
