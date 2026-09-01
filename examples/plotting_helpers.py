"""Reusable plotting helpers for 2D splitter examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np

from art.tree import LeafNode, RegressionTree, TreeNode


ContourScale = Literal["linear", "log", "symlog", "auto"]


@dataclass(frozen=True)
class TreeLeafGrid:
    """A routed 2D plotting grid shared by tree-region visualizations."""

    x1: np.ndarray
    x2: np.ndarray
    leaves: tuple[LeafNode, ...]
    labels: np.ndarray


def make_2d_grid(
    bounds: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate meshes and flattened points covering a 2D box."""

    x1, x2 = _make_2d_axes(bounds, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, points


def _make_2d_axes(
    bounds: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-dimensional coordinate axes for a regular 2D box grid."""

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 2):
        raise ValueError(f"bounds must have shape (2, 2), got {bounds.shape}.")
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")
    return (
        np.linspace(bounds[0, 0], bounds[0, 1], resolution),
        np.linspace(bounds[1, 0], bounds[1, 1], resolution),
    )


def make_tree_leaf_grid(
    tree: RegressionTree,
    bounds: np.ndarray,
    resolution: int = 300,
) -> TreeLeafGrid:
    """Route a regular 2D grid through a tree once for reuse by plots."""

    leaves = tuple(tree.iter_leaves())
    if not leaves:
        raise ValueError("Cannot plot a tree without leaves.")
    x1, x2 = _make_2d_axes(bounds, resolution)
    points = np.column_stack([np.tile(x1, x2.size), np.repeat(x2, x1.size)])
    labels = _leaf_labels(tree, points, leaves).reshape(x2.size, x1.size)
    return TreeLeafGrid(x1=x1, x2=x2, leaves=leaves, labels=labels)


def resolve_contour_scale(
    values: np.ndarray,
    scale: ContourScale,
    dynamic_range_threshold: float = 1e3,
) -> Literal["linear", "log", "symlog"]:
    """Resolve a requested contour scale from robust value magnitudes."""

    values = np.asarray(values, dtype=float)
    if scale not in ("linear", "log", "symlog", "auto"):
        raise ValueError("scale must be 'linear', 'log', 'symlog', or 'auto'.")
    if dynamic_range_threshold <= 1.0:
        raise ValueError("dynamic_range_threshold must be greater than 1.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Contour values must be finite.")
    if scale == "log" and np.any(values < 0.0):
        raise ValueError("Log contours require nonnegative values; use symlog for signed data.")
    if scale != "auto":
        return scale

    magnitudes = np.abs(values)
    nonzero = magnitudes[magnitudes > 0.0]
    if nonzero.size < 2:
        return "linear"
    low, high = np.quantile(nonzero, [0.01, 0.99])
    robust_range = float(high / max(low, np.finfo(float).tiny))
    if robust_range < dynamic_range_threshold:
        return "linear"
    return "log" if np.all(values >= 0.0) else "symlog"


def save_function_contour(
    function: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    title: str,
    out_path: Path,
    resolution: int = 300,
    levels: int = 30,
    scale: ContourScale = "linear",
    dynamic_range_threshold: float = 1e3,
    symlog_linthresh: float | None = None,
) -> Literal["linear", "log", "symlog"]:
    """Evaluate a vectorized 2D function and save its filled contour plot."""

    X1, X2, points = make_2d_grid(bounds, resolution)
    values = np.asarray(function(points), dtype=float).reshape(X1.shape)
    resolved_scale = resolve_contour_scale(
        values,
        scale=scale,
        dynamic_range_threshold=dynamic_range_threshold,
    )
    plot_values, contour_levels, norm, cmap = _contour_style(
        values,
        scale=resolved_scale,
        levels=levels,
        symlog_linthresh=symlog_linthresh,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(
        X1,
        X2,
        plot_values,
        levels=contour_levels,
        norm=norm,
        cmap=cmap,
    )
    ax.contour(
        X1,
        X2,
        plot_values,
        levels=contour_levels,
        colors="black",
        linewidths=0.25,
        alpha=0.35,
    )
    colorbar = fig.colorbar(filled, ax=ax, label="f(x)")
    if resolved_scale == "log":
        log_ticks = np.geomspace(norm.vmin, norm.vmax, min(7, levels + 1))
        colorbar.set_ticks(log_ticks, labels=[f"{tick:.1e}" for tick in log_ticks])
    ax.set(xlabel="x1", ylabel="x2", title=title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved plot: {out_path}")
    return resolved_scale


def _contour_style(
    values: np.ndarray,
    scale: Literal["linear", "log", "symlog"],
    levels: int,
    symlog_linthresh: float | None,
) -> tuple[np.ndarray, int | np.ndarray, LogNorm | SymLogNorm | None, str]:
    """Return plot values, levels, normalization, and colormap for a scale."""

    if levels < 2:
        raise ValueError("levels must be at least 2.")
    if scale == "linear":
        return values, levels, None, "viridis"

    if scale == "log":
        positive = values[values > 0.0]
        if positive.size == 0:
            raise ValueError("Log contours require at least one positive value.")
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        if vmin == vmax:
            raise ValueError("Log contours require nonconstant positive values.")
        norm = LogNorm(vmin=vmin, vmax=vmax)
        contour_levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, levels + 1)))
        return np.maximum(values, vmin), contour_levels, norm, "viridis"

    magnitudes = np.abs(values)
    nonzero = magnitudes[magnitudes > 0.0]
    if nonzero.size == 0:
        raise ValueError("Symlog contours require at least one nonzero value.")
    if symlog_linthresh is None:
        symlog_linthresh = float(np.quantile(nonzero, 0.01))
    if not np.isfinite(symlog_linthresh) or symlog_linthresh <= 0.0:
        raise ValueError("symlog_linthresh must be positive and finite.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        raise ValueError("Symlog contours require nonconstant values.")
    norm = SymLogNorm(
        linthresh=float(symlog_linthresh),
        vmin=vmin,
        vmax=vmax,
        base=10.0,
    )
    contour_levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, levels + 1)))
    return values, contour_levels, norm, "coolwarm"


def predict_tree_leaf_grid(leaf_grid: TreeLeafGrid) -> np.ndarray:
    """Evaluate each leaf model once on its assigned plotting-grid points."""

    predictions = np.empty(leaf_grid.labels.shape, dtype=float)
    for label, leaf in enumerate(leaf_grid.leaves):
        rows, columns = np.nonzero(leaf_grid.labels == label)
        if rows.size == 0:
            continue
        points = np.column_stack([leaf_grid.x1[columns], leaf_grid.x2[rows]])
        values = np.asarray(leaf.model.predict(points), dtype=float).reshape(-1)
        if values.shape[0] != rows.size:
            raise ValueError(
                f"Leaf model returned {values.shape[0]} predictions for "
                f"{rows.size} grid points."
            )
        predictions[rows, columns] = values
    return predictions


def save_tree_contour(
    leaf_grid: TreeLeafGrid,
    title: str,
    out_path: Path,
    levels: int = 30,
    scale: ContourScale = "linear",
    dynamic_range_threshold: float = 1e3,
    symlog_linthresh: float | None = None,
) -> Literal["linear", "log", "symlog"]:
    """Evaluate a routed tree grid and save its predicted filled contours."""

    values = predict_tree_leaf_grid(leaf_grid)
    resolved_scale = resolve_contour_scale(
        values,
        scale=scale,
        dynamic_range_threshold=dynamic_range_threshold,
    )
    plot_values, contour_levels, norm, cmap = _contour_style(
        values,
        scale=resolved_scale,
        levels=levels,
        symlog_linthresh=symlog_linthresh,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(
        leaf_grid.x1,
        leaf_grid.x2,
        plot_values,
        levels=contour_levels,
        norm=norm,
        cmap=cmap,
    )
    ax.contour(
        leaf_grid.x1,
        leaf_grid.x2,
        plot_values,
        levels=contour_levels,
        colors="black",
        linewidths=0.25,
        alpha=0.35,
    )
    colorbar = fig.colorbar(filled, ax=ax, label="tree prediction")
    if resolved_scale == "log":
        log_ticks = np.geomspace(norm.vmin, norm.vmax, min(7, levels + 1))
        colorbar.set_ticks(log_ticks, labels=[f"{tick:.1e}" for tick in log_ticks])
    ax.set(xlabel="x1", ylabel="x2", title=title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved plot: {out_path}")
    return resolved_scale


def save_tree_leaf_regions(
    tree: RegressionTree,
    bounds: np.ndarray,
    title: str,
    out_path: Path,
    resolution: int = 300,
    label_leaves: bool = False,
    leaf_grid: TreeLeafGrid | None = None,
) -> None:
    """Save sampled leaf regions with different colors for adjacent leaves."""

    leaf_grid = (
        make_tree_leaf_grid(tree, bounds, resolution)
        if leaf_grid is None
        else leaf_grid
    )
    colors = _adjacency_colors(leaf_grid.labels, len(leaf_grid.leaves))
    color_grid = np.asarray(
        [colors[label] for label in leaf_grid.labels.ravel()]
    ).reshape(leaf_grid.labels.shape)
    n_colors = max(colors.values()) + 1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(
        leaf_grid.x1,
        leaf_grid.x2,
        color_grid,
        shading="nearest",
        cmap=plt.get_cmap("tab20", n_colors),
        vmin=-0.5,
        vmax=n_colors - 0.5,
    )
    _draw_leaf_boundaries(
        ax,
        leaf_grid,
        bounds,
        label_leaves=label_leaves,
    )
    _finish_tree_plot(ax, bounds, title, out_path)


def save_tree_leaf_error_regions(
    tree: RegressionTree,
    bounds: np.ndarray,
    title: str,
    error_label: str,
    out_path: Path,
    resolution: int = 300,
    label_leaves: bool = False,
    leaf_grid: TreeLeafGrid | None = None,
) -> None:
    """Save leaf regions shaded by their logarithmic training fit error."""

    leaf_grid = (
        make_tree_leaf_grid(tree, bounds, resolution)
        if leaf_grid is None
        else leaf_grid
    )
    errors = np.asarray(
        [leaf.metadata.get("fit_error", np.nan) for leaf in leaf_grid.leaves],
        dtype=float,
    )
    if not np.all(np.isfinite(errors)):
        raise ValueError("Every leaf must contain a finite fit_error.")
    if np.any(errors < 0.0):
        raise ValueError("Leaf fit errors must be nonnegative.")

    plot_errors, norm = _leaf_error_style(errors)
    error_grid = plot_errors[leaf_grid.labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    shading = ax.pcolormesh(
        leaf_grid.x1,
        leaf_grid.x2,
        error_grid,
        shading="nearest",
        cmap="Greys",
        norm=norm,
    )
    colorbar = fig.colorbar(shading, ax=ax, label=error_label)
    if isinstance(norm, LogNorm):
        ticks = np.geomspace(norm.vmin, norm.vmax, 7)
        colorbar.set_ticks(ticks, labels=[f"{tick:.1e}" for tick in ticks])
    elif np.all(errors == 0.0):
        colorbar.set_ticks([0.0], labels=["0"])
    _draw_leaf_boundaries(
        ax,
        leaf_grid,
        bounds,
        label_leaves=label_leaves,
    )
    _finish_tree_plot(ax, bounds, title, out_path)


def _leaf_error_style(
    errors: np.ndarray,
) -> tuple[np.ndarray, LogNorm | Normalize]:
    """Floor zero errors and construct a stable logarithmic normalization."""

    positive = errors[errors > 0.0]
    if positive.size == 0:
        return errors, Normalize(vmin=0.0, vmax=1.0)

    minimum = float(np.min(positive))
    maximum = float(np.max(positive))
    if minimum == maximum:
        vmin = minimum / 10.0
        vmax = maximum * 10.0
    else:
        vmin = minimum / 10.0 if np.any(errors == 0.0) else minimum
        vmax = maximum
    return np.maximum(errors, vmin), LogNorm(vmin=vmin, vmax=vmax)


def _draw_leaf_boundaries(
    ax: plt.Axes,
    leaf_grid: TreeLeafGrid,
    bounds: np.ndarray,
    label_leaves: bool,
) -> None:
    """Draw sampled leaf outlines and optional nonoverlapping labels."""

    span = bounds[:, 1] - bounds[:, 0]
    min_label_distance = 0.06 * float(np.linalg.norm(span))
    label_margin = 0.025 * span
    label_positions: list[tuple[float, float]] = []
    for label in range(len(leaf_grid.leaves)):
        mask = leaf_grid.labels == label
        if np.any(mask) and not np.all(mask):
            ax.contour(
                leaf_grid.x1,
                leaf_grid.x2,
                mask.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.8,
            )
        if label_leaves and np.any(mask):
            rows, columns = np.nonzero(mask)
            position = np.clip(
                [np.mean(leaf_grid.x1[columns]), np.mean(leaf_grid.x2[rows])],
                bounds[:, 0] + label_margin,
                bounds[:, 1] - label_margin,
            )
            is_separated = all(
                np.linalg.norm(np.subtract(position, other)) >= min_label_distance
                for other in label_positions
            )
            if is_separated:
                ax.text(*position, f"L{label}", ha="center", va="center", fontsize=7, color="black")
                label_positions.append((float(position[0]), float(position[1])))


def _finish_tree_plot(
    ax: plt.Axes,
    bounds: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Apply common tree-plot formatting and save the figure."""

    fig = ax.figure
    ax.set(xlabel="x1", ylabel="x2", title=title)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def _leaf_labels(
    tree: RegressionTree,
    X: np.ndarray,
    leaves: tuple[LeafNode, ...],
) -> np.ndarray:
    """Route point batches recursively and return integer leaf labels."""

    if tree.root is None:
        raise ValueError("Tree has no root node.")
    labels_by_id = {id(leaf): label for label, leaf in enumerate(leaves)}
    labels = np.empty(X.shape[0], dtype=int)

    def assign(node: TreeNode, indices: np.ndarray) -> None:
        if isinstance(node, LeafNode):
            labels[indices] = labels_by_id[id(node)]
            return
        right = (X[indices] @ node.w - node.z) >= 0.0
        if np.any(~right):
            assign(node.left, indices[~right])
        if np.any(right):
            assign(node.right, indices[right])

    assign(tree.root, np.arange(X.shape[0]))
    return labels


def _adjacency_colors(labels: np.ndarray, n_labels: int) -> dict[int, int]:
    """Greedily color the adjacency graph induced by a labeled grid."""

    neighbors = [set() for _ in range(n_labels)]
    horizontal = np.column_stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()])
    vertical = np.column_stack([labels[:-1, :].ravel(), labels[1:, :].ravel()])
    for first, second in np.vstack([horizontal, vertical]):
        if first != second:
            neighbors[int(first)].add(int(second))
            neighbors[int(second)].add(int(first))

    assigned: dict[int, int] = {}
    order = sorted(range(n_labels), key=lambda label: len(neighbors[label]), reverse=True)
    for label in order:
        unavailable = {assigned[neighbor] for neighbor in neighbors[label] if neighbor in assigned}
        color = 0
        while color in unavailable:
            color += 1
        assigned[label] = color
    return assigned


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
                lower = float(np.min(positive))
                upper = float(np.max(positive))
                if lower == upper:
                    lower /= 10.0
                    upper *= 10.0
                bins_arg = np.logspace(np.log10(lower), np.log10(upper), bins + 1)
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
