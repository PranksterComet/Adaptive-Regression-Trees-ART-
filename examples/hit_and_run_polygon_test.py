"""Visual smoke test for hit-and-run sampling on a 2D polygon."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import PolytopeRegion
from art.sampling import HitAndRunSampler


def regular_polygon_region(n_sides: int = 6, radius: float = 1.0) -> tuple[PolytopeRegion, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    vertices = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    A = []
    b = []
    for i in range(n_sides):
        p = vertices[i]
        q = vertices[(i + 1) % n_sides]
        edge = q - p
        normal = np.array([edge[1], -edge[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        bound = float(normal @ p)
        A.append(normal)
        b.append(bound)

    return PolytopeRegion(np.asarray(A), np.asarray(b), tag="regular_polygon"), vertices


def grid_uniformity_score(
    region: PolytopeRegion,
    samples: np.ndarray,
    bounds: np.ndarray,
    grid_size: int = 20,
) -> dict[str, float]:
    """Approximate uniformity by comparing sample counts to grid-cell area estimates."""

    x_edges = np.linspace(bounds[0, 0], bounds[0, 1], grid_size + 1)
    y_edges = np.linspace(bounds[1, 0], bounds[1, 1], grid_size + 1)
    counts, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges])

    centers_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centers_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(centers_x, centers_y, indexing="ij")
    centers = np.column_stack([Xc.ravel(), Yc.ravel()])
    inside = region.contains(centers).reshape(grid_size, grid_size)

    observed = counts[inside]
    expected = np.full(observed.shape, samples.shape[0] / observed.size)
    chi_square = np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-12))
    coefficient_of_variation = np.std(observed) / max(np.mean(observed), 1e-12)

    return {
        "occupied_grid_cells": float(observed.size),
        "grid_count_chi_square": float(chi_square),
        "grid_count_cv": float(coefficient_of_variation),
    }


def main() -> None:
    region, vertices = regular_polygon_region(n_sides=7, radius=1.0)
    bounds = np.array([[-1.05, 1.05], [-1.05, 1.05]])

    sampler = HitAndRunSampler(burn_in=1000, thinning=1, bounds=bounds)
    samples = sampler.sample(region, n=5000, random_state=4, x0=np.array([0.0, 0.0]))

    diagnostics = grid_uniformity_score(region, samples, bounds=bounds, grid_size=20)
    diagnostics["sample_mean_norm"] = float(np.linalg.norm(np.mean(samples, axis=0)))

    for key, value in diagnostics.items():
        print(f"{key}: {value:.6g}")

    closed_vertices = np.vstack([vertices, vertices[0]])
    plt.figure(figsize=(6, 6))
    plt.plot(closed_vertices[:, 0], closed_vertices[:, 1], color="black", linewidth=2)
    plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.25)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Hit-and-Run Samples in a 2D Polygon")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()

    out_path = Path(__file__).with_name("hit_and_run_polygon_samples.png")
    plt.savefig(out_path, dpi=200)
    print(f"saved plot: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
