from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from examples.tree_shallow_water_benchmark import (
    DEFAULT_N_TEST,
    make_shallow_water_oracle,
    main,
    parse_args,
    validate_args,
)


def test_shallow_water_harness_constructs_configured_oracle() -> None:
    args = parse_args(
        [
            "--eta-left-range",
            "1.6",
            "2.8",
            "--eta-right-range",
            "0.7",
            "1.1",
            "--bump-height-range",
            "0.2",
            "0.5",
            "--sensor-position",
            "2.5",
            "--final-time",
            "0.6",
            "--cells",
            "200",
        ]
    )
    oracle, metadata = make_shallow_water_oracle(args)

    assert args.n_test == DEFAULT_N_TEST
    np.testing.assert_allclose(
        oracle.parameter_bounds,
        [[1.6, 2.8], [0.7, 1.1], [0.2, 0.5]],
    )
    np.testing.assert_allclose(
        oracle.map_to_physical(np.zeros(3)),
        [2.2, 0.9, 0.35],
    )
    assert oracle.sensor_position == 2.5
    assert oracle.final_time == 0.6
    assert oracle.cells == 200
    assert metadata["quantity_of_interest"] == "final water depth at sensor"


def test_shallow_water_harness_validates_harness_arguments() -> None:
    args = parse_args(["--n-test", "0"])
    with pytest.raises(ValueError, match="--n-test"):
        validate_args(args)

    args = parse_args(["--splitter", "hrt", "--leaf-degree", "2"])
    with pytest.raises(ValueError, match="requires --leaf-degree 1"):
        validate_args(args)


def test_shallow_water_harness_exposes_min_side_fallback() -> None:
    args = parse_args(["--min-side-fallback"])

    assert args.min_side_fallback is True


def test_shallow_water_harness_exposes_parallel_sampler() -> None:
    args = parse_args(["--sampler", "parallel_numba"])

    assert args.sampler == "parallel_numba"
    assert args.n_chains == 50
    assert args.sampler_threads == 10
    assert args.parallel_random


@pytest.mark.skipif(
    importlib.util.find_spec("clawpack") is None,
    reason="Clawpack is only installed in the art Conda environment.",
)
@pytest.mark.parametrize("space_time", [False, True])
def test_shallow_water_harness_smoke_run(tmp_path, space_time) -> None:
    output_dir = tmp_path / "shallow_water"
    main(
        [
            "--output-dir",
            str(output_dir),
            "--cells",
            "50",
            "--sample-count",
            "4",
            "--max-depth",
            "0",
            "--error-tolerance",
            "1e6",
            "--n-test",
            "3",
            "--test-batch-size",
            "2",
            "--no-isotropic-sampling",
        ] + (["--space-time"] if space_time else [])
    )

    assert (output_dir / "tree.joblib").is_file()
    assert (output_dir / "node_diagnostics.csv").is_file()
    assert (output_dir / "test_error_histogram.png").is_file()
    report = (output_dir / "report.txt").read_text()
    assert "mean_test_oracle_query:" in report
    assert "physical_parameter_bounds:" in report
    if space_time:
        assert "dimension: 5" in report
        assert "water depth at input position and time" in report


def test_space_time_harness_metadata() -> None:
    args = parse_args(["--space-time", "--position-range", "0", "4",
                       "--time-range", "0", "0.5"])
    oracle, metadata = make_shallow_water_oracle(args)
    assert oracle.dimension == 5
    np.testing.assert_allclose(oracle.parameter_bounds[-2:], [[0, 4], [0, 0.5]])
    assert "sensor_position" not in metadata
    assert "final_time" not in metadata
