from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from oracles.shallow_water import ShallowWaterSensorOracle


def test_shallow_water_oracle_maps_normalized_to_physical_parameters() -> None:
    oracle = ShallowWaterSensorOracle()

    assert oracle.parameter_names == (
        "eta_left",
        "eta_right",
        "bump_height",
    )
    assert np.array_equal(
        oracle.parameter_bounds,
        [[1.5, 3.0], [0.6, 1.2], [0.1, 0.4]],
    )
    assert np.allclose(
        oracle.map_to_physical(np.array([-1.0, -1.0, -1.0])),
        [1.5, 0.6, 0.1],
    )
    assert np.allclose(
        oracle.map_to_physical(np.zeros(3)),
        [2.25, 0.9, 0.25],
    )
    assert np.allclose(
        oracle.map_to_physical(np.ones((2, 3))),
        [[3.0, 1.2, 0.4], [3.0, 1.2, 0.4]],
    )


def test_shallow_water_oracle_validates_domain_and_inputs() -> None:
    with pytest.raises(ValueError, match="eta_left > eta_right"):
        ShallowWaterSensorOracle(
            eta_left_range=(1.0, 2.0),
            eta_right_range=(0.8, 1.2),
        )
    with pytest.raises(ValueError, match="fully wet"):
        ShallowWaterSensorOracle(
            eta_right_range=(0.3, 1.2),
            bump_height_range=(0.1, 0.4),
        )

    oracle = ShallowWaterSensorOracle()
    with pytest.raises(ValueError, match="normalized box"):
        oracle(np.array([1.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        oracle(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="outside"):
        oracle.evaluate_physical(np.array([1.4, 1.0, 0.2]))


@pytest.mark.skipif(
    importlib.util.find_spec("clawpack") is None,
    reason="Clawpack is only installed in the art Conda environment.",
)
def test_shallow_water_oracle_scalar_and_batch_evaluation() -> None:
    oracle = ShallowWaterSensorOracle(cells=100)
    first = np.array([0.0, 0.0, 0.0])
    second = np.array([0.5, -0.5, -0.5])

    scalar = oracle(first)
    batch = oracle(np.vstack([first, second]))

    assert isinstance(scalar, float)
    assert batch.shape == (2,)
    assert np.isclose(batch[0], scalar)
    assert np.all(np.isfinite(batch))
    assert np.all(batch > 0.0)


def test_space_time_parameter_domain() -> None:
    oracle = ShallowWaterSensorOracle(
        sensor_position_range=(0.0, 5.0), time_range=(0.0, 0.75)
    )
    assert oracle.dimension == 5
    assert oracle.parameter_names[-2:] == ("position", "time")
    np.testing.assert_allclose(
        oracle.map_to_physical(np.array([-np.ones(5), np.zeros(5), np.ones(5)])),
        [[1.5, 0.6, 0.1, 0.0, 0.0],
         [2.25, 0.9, 0.25, 2.5, 0.375],
         [3.0, 1.2, 0.4, 5.0, 0.75]],
    )
    assert oracle(np.empty((0, 5))).shape == (0,)
    with pytest.raises(ValueError, match="both"):
        ShallowWaterSensorOracle(sensor_position_range=(0.0, 5.0))
    with pytest.raises(ValueError, match="nonnegative"):
        ShallowWaterSensorOracle(sensor_position_range=(0.0, 5.0), time_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="outer cell centers"):
        ShallowWaterSensorOracle(sensor_position_range=(0.0, 10.0), time_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="outside"):
        oracle.evaluate_physical([2.0, 1.0, 0.3, 3.0, -0.1])


@pytest.mark.skipif(importlib.util.find_spec("clawpack") is None, reason="Requires Clawpack")
def test_space_time_matches_fixed_sensor_oracle_and_initial_condition() -> None:
    oracle = ShallowWaterSensorOracle(
        cells=100, sensor_position_range=(0.0, 5.0), time_range=(0.0, 0.75)
    )
    physics = [2.0, 1.0, 0.3]
    # Mix observation times and physical parameters to detect state carry-over.
    points = np.array([[*physics, 3.0, 0.75], [2.5, 0.8, 0.2, 0.0, 0.1],
                       [*physics, 5.0, 0.0], [*physics, 0.0, 0.0],
                       [*physics, 3.0, 0.75]])
    values = oracle.evaluate_physical(points)
    for i in (0, 1):
        fixed = ShallowWaterSensorOracle(
            cells=100, sensor_position=points[i, 3], final_time=points[i, 4]
        )
        assert values[i] == pytest.approx(fixed.evaluate_physical(points[i, :3]), abs=1e-12)
    assert values[0] == values[-1]
    x = np.linspace(-9.9, 9.9, 100)
    initial = np.where(x < 0.0, 2.0, 1.0) - 0.3 * np.exp(-((x - 2.0) / 0.75)**2)
    np.testing.assert_allclose(values[2:4], np.interp([5.0, 0.0], x, initial), atol=1e-12)
