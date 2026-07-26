from __future__ import annotations

import numpy as np

from examples.plotting_helpers import _leaf_error_style, resolve_contour_scale


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
