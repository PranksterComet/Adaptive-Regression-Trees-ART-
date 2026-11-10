"""Check oracle parity with recorded ART runs, independent of grid accuracy."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tasmanian_comparison", ROOT / "tasmanian/tasmanian_comparison.py"
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
from oracle_config import PRESETS, make_oracle


def read_report(path):
    sections = {}
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections[section] = {}
        elif section and ": " in line:
            key, value = line.split(": ", 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            sections[section][key] = value
    return sections


@pytest.mark.parametrize("name,dimension,preset", [
    (name, int(d), preset) for name, dimensions in PRESETS.items() for d, preset in dimensions.items()
])
def test_presets_match_source_reports(name, dimension, preset):
    source = ROOT / preset["source_report"]
    if not source.exists():
        pytest.skip("Local TACC results are not checked into the repository")
    report = read_report(source)
    assert preset["target"] == json.loads(json.dumps(report["target"]))
    assert preset["bounds"] == report["configuration"]["resolved_domain_bounds"]
    _, bounds, metadata = make_oracle(example.parse_args(["--benchmark", name, "--dimension", str(dimension)]))
    np.testing.assert_array_equal(bounds, preset["bounds"])
    for key, value in preset["target"].items():
        actual = metadata["parameters"][key]
        if isinstance(value, (float, int, list, tuple)):
            if key == "parameter_names":
                assert list(actual) == list(value)
            else:
                np.testing.assert_allclose(actual, value, rtol=1e-14, atol=1e-14)
        else:
            assert actual == value


SYNTHETIC_REPORTS = [p for p in sorted((ROOT / "results").rglob("report.txt")) if "shallow_water" not in p.parts]


@pytest.mark.parametrize("path", SYNTHETIC_REPORTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_recorded_synthetic_oracle(path):
    report = read_report(path)
    c, t = report["configuration"], report["target"]
    d, name = c["dimension"], c["benchmark"]
    bounds = np.asarray(c["resolved_domain_bounds"])
    flags = ["--benchmark", name, "--dimension", str(d),
             "--domain-low", str(bounds[0, 0]), "--domain-high", str(bounds[0, 1])]
    if name == "plane_wave" and t["normal"][0] == 1.0:
        flags += ["--normal", "axis"]
    oracle, resolved_bounds, _ = make_oracle(example.parse_args(flags))
    np.testing.assert_array_equal(resolved_bounds, bounds)
    x = np.random.default_rng(23).uniform(bounds[:, 0], bounds[:, 1], (40, d))
    if name == "rosenbrock":
        y = np.sum(t["b"] * (x[:, 1:] - x[:, :-1]**2)**2 + (t["a"] - x[:, :-1])**2, axis=1)
    elif name == "rastrigin":
        y = t["A"] * d + np.sum(x*x - t["A"]*np.cos(2*np.pi*x), axis=1)
    elif name == "plane_wave":
        y = t["beta"] + t["amplitude"] * np.cos(t["frequency"] * (x @ np.asarray(t["normal"])) + t["phase"])
    else:
        # Include points on both sides of the exact saved radius.
        x[:2] = 0
        x[0, 0], x[1, 0] = t["radius"] - 1e-8, t["radius"] + 1e-8
        inside = np.sum((x-np.asarray(t["center"]))**2, axis=1) <= t["radius"]**2
        y = np.where(inside, t["inside_beta"] + x @ np.asarray(t["inside_linear_coefficients"]),
                     t["outside_beta"] + x @ np.asarray(t["outside_linear_coefficients"]))
    np.testing.assert_allclose(oracle(x), y + t["offset"], rtol=1e-13, atol=1e-12)
    assert oracle(x[0]) == pytest.approx(y[0] + t["offset"])


@pytest.mark.parametrize("flags", [
    ["--benchmark", "shallow_water_sensor", "--dimension", "5"],
    ["--benchmark", "shallow_water_space_time", "--sensor-position", "3"],
    ["--benchmark", "shallow_water_sensor", "--domain-low", "0", "--domain-high", "1"],
    ["--benchmark", "rosenbrock", "--frequency", "4"],
    ["--domain-low", "-1"],
    ["--domain-low", "1", "--domain-high", "-1"],
    ["--benchmark", "plane_wave", "--dimension", "4"],
    ["--benchmark", "spherical_piecewise", "--dimension", "6"],
    ["--benchmark", "rosenbrock", "--active-dimension", "3"],
    ["--benchmark", "active_subspace_gaussian", "--domain-low", "-1", "--domain-high", "1"],
])
def test_rejects_invalid_or_ignored_settings(flags):
    with pytest.raises(ValueError):
        make_oracle(example.parse_args(flags))


def test_defaults_and_explicit_zero_offset():
    oracle, bounds, info = make_oracle(example.parse_args([]))
    assert info["dimension"] == 5
    assert oracle(np.ones(5)) == 1.0
    assert bounds.shape == (5, 2)
    oracle, _, info = make_oracle(example.parse_args(["--dimension", "2", "--offset", "0"]))
    assert oracle(np.ones(2)) == 0.0
    assert info["notes"]
    for d, half_width in [(3, 5.12), (4, 2.5), (5, 1.5)]:
        _, bounds, _ = make_oracle(example.parse_args(["--benchmark", "rastrigin", "--dimension", str(d)]))
        np.testing.assert_array_equal(bounds, np.tile([-half_width, half_width], (d, 1)))


def test_spherical_offset_overrides_the_fixed_affine_default():
    oracle, _, info = make_oracle(
        example.parse_args(
            [
                "--benchmark",
                "spherical_piecewise",
                "--dimension",
                "3",
                "--offset",
                "1.25",
            ]
        )
    )

    assert oracle.offset == pytest.approx(1.25)
    assert info["parameters"]["offset"] == pytest.approx(1.25)

    default_oracle, _, default_info = make_oracle(
        example.parse_args(
            ["--benchmark", "spherical_piecewise", "--dimension", "3"]
        )
    )
    assert default_oracle.offset == pytest.approx(4.15)
    assert default_info["parameters"]["offset"] == pytest.approx(4.15)


def test_active_subspace_gaussian_defaults_and_overrides():
    oracle, bounds, info = make_oracle(
        example.parse_args(["--benchmark", "active_subspace_gaussian"])
    )
    target = oracle.target

    assert info["dimension"] == 5
    assert target.active_dimension == 3
    assert target.eta == 0.1
    assert target.a == 1.0
    assert oracle.offset == 1.0
    np.testing.assert_allclose(target.tau, [0.2, 0.5, 0.8])
    np.testing.assert_allclose(target.active_mean, np.zeros(3))
    np.testing.assert_allclose(target.U.T @ target.U, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(bounds, np.tile([-1.0, 1.0], (5, 1)))
    assert oracle(np.zeros(5)) == pytest.approx(2.0)

    customized, custom_bounds, _ = make_oracle(
        example.parse_args(
            [
                "--benchmark", "active_subspace_gaussian",
                "--dimension", "4",
                "--active-dimension", "2",
                "--eta", "0.25",
                "--a", "2",
                "--tau", "0.3", "0.7",
                "--active-mean", "0.1", "-0.2",
                "--seed", "11",
                "--offset", "0",
            ]
        )
    )
    assert customized.target.active_dimension == 2
    assert customized.target.eta == 0.25
    np.testing.assert_allclose(customized.target.tau, [0.3, 0.7])
    np.testing.assert_allclose(customized.target.active_mean, [0.1, -0.2])
    np.testing.assert_array_equal(custom_bounds, np.tile([-2.0, 2.0], (4, 1)))
    assert customized.offset == 0.0


def test_error_summary_matches_art_report_metrics():
    truth = np.array([1.0, 2.0, 4.0, 8.0])
    prediction = np.array([0.5, 2.5, 3.0, 10.0])

    performance, quantiles = example.error_summary(truth, prediction, 1e-12)
    residual = np.abs(truth - prediction)
    relative = residual / np.abs(truth)

    assert performance["test_mse"] == pytest.approx(np.mean(residual**2))
    assert performance["test_rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert performance["test_relative_l2_error"] == pytest.approx(
        np.linalg.norm(residual) / np.linalg.norm(truth)
    )
    assert performance["test_median_pointwise_relative_error"] == pytest.approx(
        np.median(relative)
    )
    assert performance["test_max_pointwise_relative_error"] == np.max(relative)
    assert performance["test_max_absolute_error"] == np.max(residual)
    assert tuple(quantiles) == (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def test_text_report_uses_art_sections_and_quantiles(tmp_path):
    args = example.parse_args([])
    _, bounds, target = make_oracle(args)
    args.dimension = target["dimension"]
    truth = np.array([1.0, 2.0, 4.0])
    performance, quantiles = example.error_summary(
        truth,
        np.array([1.1, 1.8, 4.4]),
        args.relative_error_floor,
    )
    report = example.format_report(
        args=args,
        output_dir=tmp_path,
        bounds=bounds,
        target_metadata=target,
        tasmanian_version="test-version",
        stop_reason="surplus_tolerance",
        training_queries=17,
        n_iterations=3,
        performance=performance,
        relative_quantiles=quantiles,
        truth=truth,
        timings={"grid_build": 1.0, "test_oracle_evaluation": 0.2},
    )

    sections = set()
    for line in report.splitlines():
        if line.startswith("[") and line.endswith("]"):
            sections.add(line)
    assert sections == {
        "[configuration]",
        "[target]",
        "[performance]",
        "[test_oracle_range]",
        "[pointwise_relative_error_quantiles]",
        "[sparse_grid]",
        "[timing_seconds]",
    }
    assert "test_relative_l2_error:" in report
    assert "test_median_pointwise_relative_error:" in report
    assert "q99.9:" in report
    assert "training_queries: 17" in report


@pytest.mark.parametrize("space_time", [False, True])
def test_water_matches_existing_oracle(space_time):
    pytest.importorskip("clawpack.pyclaw")
    from oracles.shallow_water import ShallowWaterSensorOracle
    name = "shallow_water_space_time" if space_time else "shallow_water_sensor"
    oracle, bounds, info = make_oracle(example.parse_args(["--benchmark", name]))
    reference = ShallowWaterSensorOracle(sensor_position_range=(0, 5) if space_time else None,
                                         time_range=(0, .75) if space_time else None)
    x = np.array([[-1.] * reference.dimension, [0.] * reference.dimension, [1.] * reference.dimension])
    np.testing.assert_allclose(oracle(x), reference(x), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(bounds, np.tile([-1, 1], (reference.dimension, 1)))
    assert info["parameters"]["offset"] == 0
    assert info["parameters"]["cells"] == 500
    assert bool(info["notes"]) == (not space_time)


@pytest.mark.parametrize("name", ["active_subspace_gaussian", "rosenbrock", "rastrigin", "plane_wave", "spherical_piecewise",
                                  "shallow_water_sensor", "shallow_water_space_time"])
def test_runner_saves_resolved_settings_and_reloadable_grid(tmp_path, name):
    Tasmanian = pytest.importorskip("Tasmanian")
    if name.startswith("shallow_water"):
        pytest.importorskip("clawpack.pyclaw")
    out = tmp_path / name
    flags = ["--benchmark", name, "--initial-depth", "0", "--max-iterations", "1",
             "--n-test", "3", "--batch-size", "2", "--output-dir", str(out)]
    if name == "active_subspace_gaussian":
        flags += ["--n-gaussian-test", "3"]
    example.main(flags)
    report = read_report(out / "report.txt")
    assert report["target"]["benchmark"] == name
    assert report["target"]["preset_source_report"]
    assert "test_relative_l2_error" in report["performance"]
    assert "q99.9" in report["pointwise_relative_error_quantiles"]
    if name == "active_subspace_gaussian":
        assert "test_relative_l2_error" in report["gaussian_weighted_performance"]
        assert "q99.9" in report["gaussian_weighted_pointwise_relative_error_quantiles"]
    grid = Tasmanian.SparseGrid()
    grid.read(str(out / "grid.tsgrid"))
    saved = np.load(out / "test_predictions.npz")
    np.testing.assert_allclose(grid.evaluateBatch(saved["points"]).ravel(), saved["prediction"])
