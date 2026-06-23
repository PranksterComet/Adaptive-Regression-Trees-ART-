"""Example test for the HingeAffineSplitter on 2D hinge functions."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain
from art.models import AffineRidgeModel, augment_features
from art.sampling import sample_uniform_box
from art.splitters import HingeAffineSplitter


def affine_values(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return augment_features(X) @ theta


def boundary_from_thetas(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, float]:
    delta = theta1 - theta2
    w = delta[:-1].copy()
    z = -float(delta[-1])
    w_norm = np.linalg.norm(w)
    if w_norm <= 1e-12:
        raise ValueError("Degenerate boundary.")
    return w / w_norm, z / w_norm


def clipped_boundary_segment(bounds: np.ndarray, w: np.ndarray, z: float) -> np.ndarray | None:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    corners = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    points = []

    for i, j in edges:
        a = corners[i]
        b = corners[j]
        direction = b - a
        denom = float(w @ direction)
        if abs(denom) <= 1e-14:
            continue
        t = (z - float(w @ a)) / denom
        if -1e-12 <= t <= 1.0 + 1e-12:
            p = a + np.clip(t, 0.0, 1.0) * direction
            if not any(np.linalg.norm(p - q) < 1e-9 for q in points):
                points.append(p)

    if len(points) < 2:
        return None
    return np.asarray(points[:2])


def plot_boundary(
    X: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    true_w: np.ndarray,
    true_z: float,
    learned_w: np.ndarray,
    learned_z: float,
    title: str,
    out_path: Path,
) -> None:
    true_segment = clipped_boundary_segment(bounds, true_w, true_z)
    learned_segment = clipped_boundary_segment(bounds, learned_w, learned_z)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(X[:, 0], X[:, 1], c=y, s=18, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="f(x)")

    if true_segment is not None:
        plt.plot(true_segment[:, 0], true_segment[:, 1], color="white", linewidth=4)
        plt.plot(true_segment[:, 0], true_segment[:, 1], color="black", linewidth=2, label="true boundary")
    if learned_segment is not None:
        plt.plot(
            learned_segment[:, 0],
            learned_segment[:, 1],
            color="green",
            linestyle="--",
            linewidth=2.5,
            label="learned boundary",
        )

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved plot: {out_path}")
    plt.show()


def run_case(mode: str, bounds: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> None:
    X = sample_uniform_box(bounds, n=300, random_state=7)
    y1 = affine_values(X, theta1)
    y2 = affine_values(X, theta2)
    y = np.maximum(y1, y2) if mode == "max" else np.minimum(y1, y2)

    parent_model = AffineRidgeModel(ridge=1e-8).fit(X, y)

    splitter = HingeAffineSplitter(
        mode=mode,
        ridge=1e-8,
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
