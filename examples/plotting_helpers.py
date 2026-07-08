"""Reusable plotting helpers for 2D splitter examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def boundary_from_thetas(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized (w, z) for the line where two affine models agree."""

    delta = np.asarray(theta1, dtype=float) - np.asarray(theta2, dtype=float)
    w = delta[:-1].copy()
    z = -float(delta[-1])
    w_norm = np.linalg.norm(w)
    if w_norm <= 1e-12:
        raise ValueError("Degenerate boundary.")
    return w / w_norm, z / w_norm


def clipped_boundary_segment(bounds: np.ndarray, w: np.ndarray, z: float) -> np.ndarray | None:
    """Clip the infinite line w^T x = z to a 2D box for plotting."""

    bounds = np.asarray(bounds, dtype=float)
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

    # Intersect the line with each box edge and keep intersections on the edge.
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
    learned_color: str = "green",
    show: bool = True,
) -> None:
    """Plot sampled values plus true and learned 2D hyperplane boundaries."""

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
            color=learned_color,
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
    if show:
        plt.show()
    else:
        plt.close()


def save_histogram(
    values: list[float] | np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
    logy: bool = False,
    log_bins: bool = False,
) -> None:
    """Save a histogram for scalar experiment diagnostics."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    plt.figure(figsize=(7, 4))
    if values.size:
        if log_bins:
            positive = values[values > 0.0]
            if positive.size:
                bins_arg = np.logspace(np.log10(np.min(positive)), np.log10(np.max(positive)), bins + 1)
                values = positive
                plt.xscale("log")
            else:
                bins_arg = bins
        else:
            bins_arg = bins
        plt.hist(values, bins=bins_arg, edgecolor="black", alpha=0.8)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("frequency")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def save_bar_counts(
    labels: list[str],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """Save a bar chart of categorical counts."""

    counts = {label: labels.count(label) for label in sorted(set(labels))}
    plt.figure(figsize=(7, 4))
    if counts:
        plt.bar(list(counts.keys()), list(counts.values()), edgecolor="black", alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("frequency")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")
