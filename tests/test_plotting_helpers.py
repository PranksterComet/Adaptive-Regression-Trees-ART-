from __future__ import annotations

import numpy as np

from art.tree import LeafNode, RegressionTree, SplitNode
from examples.plotting_helpers import (
    _leaf_error_style,
    make_tree_leaf_grid,
    predict_tree_leaf_grid,
    resolve_contour_scale,
)


class ConstantCountingModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.batch_sizes: list[int] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(X.shape[0])
        return np.full(X.shape[0], self.value)


def test_contour_scale_manual_modes() -> None:
    values = np.array([1.0, 10.0])

    assert resolve_contour_scale(values, "linear") == "linear"
    assert resolve_contour_scale(values, "log") == "log"
    assert resolve_contour_scale(values, "symlog") == "symlog"


def test_contour_scale_auto_uses_robust_dynamic_range() -> None:
    low_range = np.linspace(1.0, 10.0, 1000)
    positive_high_range = np.geomspace(1e-6, 1e3, 1000)
    signed_high_range = np.concatenate([-positive_high_range, positive_high_range])

    assert resolve_contour_scale(low_range, "auto") == "linear"
    assert resolve_contour_scale(positive_high_range, "auto") == "log"
    assert resolve_contour_scale(signed_high_range, "auto") == "symlog"


def test_log_contours_reject_negative_values() -> None:
    try:
        resolve_contour_scale(np.array([-1.0, 2.0]), "log")
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("Expected log contours to reject negative values.")


def test_leaf_error_style_handles_zero_and_constant_errors() -> None:
    values, norm = _leaf_error_style(np.array([0.0, 1e-3, 1.0]))
    constant_values, constant_norm = _leaf_error_style(np.array([0.5, 0.5]))
    zero_values, zero_norm = _leaf_error_style(np.zeros(2))

    assert np.all(values > 0.0)
    assert norm.vmin < norm.vmax
    assert np.all(constant_values == 0.5)
    assert constant_norm.vmin < 0.5 < constant_norm.vmax
    assert np.array_equal(zero_values, np.zeros(2))
    assert zero_norm.vmin == 0.0
    assert zero_norm.vmax == 1.0


def test_tree_leaf_grid_stores_axes_and_predicts_each_leaf_once() -> None:
    left_model = ConstantCountingModel(-1.0)
    right_model = ConstantCountingModel(2.0)
    tree = RegressionTree(
        SplitNode(
            w=np.array([1.0, 0.0]),
            z=0.0,
            left=LeafNode(left_model, node_id="root/L"),
            right=LeafNode(right_model, node_id="root/R"),
            node_id="root",
        )
    )

    grid = make_tree_leaf_grid(tree, np.array([[-1.0, 1.0], [-2.0, 2.0]]), resolution=3)
    predictions = predict_tree_leaf_grid(grid)

    assert grid.x1.shape == (3,)
    assert grid.x2.shape == (3,)
    assert grid.labels.shape == (3, 3)
    np.testing.assert_array_equal(predictions[:, 0], np.full(3, -1.0))
    np.testing.assert_array_equal(predictions[:, 1:], np.full((3, 2), 2.0))
    assert left_model.batch_sizes == [3]
    assert right_model.batch_sizes == [6]
