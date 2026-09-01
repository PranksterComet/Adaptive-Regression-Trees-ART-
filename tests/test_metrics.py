from __future__ import annotations

import numpy as np
import pytest

from art.metrics import (
    pointwise_relative_error,
    pointwise_relative_error_quantile,
)


def test_pointwise_relative_error_quantile_accepts_scalar_and_vector_quantiles() -> None:
    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([0.5, 2.0, 6.0, 10.0])
    errors = pointwise_relative_error(y_true, y_pred)

    assert pointwise_relative_error_quantile(y_true, y_pred, 0.5) == pytest.approx(
        np.quantile(errors, 0.5)
    )
    np.testing.assert_allclose(
        pointwise_relative_error_quantile(y_true, y_pred, [0.0, 0.9, 1.0]),
        np.quantile(errors, [0.0, 0.9, 1.0]),
    )


@pytest.mark.parametrize("quantile", [-0.1, 1.1, np.nan])
def test_pointwise_relative_error_quantile_rejects_invalid_quantiles(
    quantile: float,
) -> None:
    with pytest.raises(ValueError, match="quantile"):
        pointwise_relative_error_quantile(np.ones(2), np.ones(2), quantile)
