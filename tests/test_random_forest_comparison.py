from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np

from random_forest.forest_comparison import main, make_forest, parse_args


def _read_sections(path: Path) -> set[str]:
    return {
        line
        for line in path.read_text().splitlines()
        if line.startswith("[") and line.endswith("]")
    }


def test_random_forest_defaults_define_fixed_baseline() -> None:
    args = parse_args([])
    forest = make_forest(args)

    assert args.training_points == 10_000
    assert forest.n_estimators == 500
    assert forest.criterion == "squared_error"
    assert forest.max_depth is None
    assert forest.min_samples_split == 2
    assert forest.min_samples_leaf == 1
    assert forest.max_features == 1.0
    assert forest.bootstrap
    assert forest.max_samples is None
    assert forest.ccp_alpha == 0.0
    assert forest.n_jobs == 1
    assert forest.random_state == 1


def test_random_forest_feature_subsampling_parser_distinguishes_count_and_fraction() -> None:
    one_feature = parse_args(["--max-features", "1"])
    all_features = parse_args(["--max-features", "1.0"])
    square_root = parse_args(["--max-features", "sqrt"])

    assert one_feature.max_features == 1
    assert isinstance(one_feature.max_features, int)
    assert all_features.max_features == 1.0
    assert isinstance(all_features.max_features, float)
    assert square_root.max_features == "sqrt"


def test_random_forest_runner_saves_reloadable_art_style_results(tmp_path) -> None:
    output_dir = tmp_path / "active_gaussian"
    main(
        [
            "--benchmark",
            "active_subspace_gaussian",
            "--dimension",
            "4",
            "--active-dimension",
            "2",
            "--training-points",
            "40",
            "--n-estimators",
            "3",
            "--n-test",
            "12",
            "--n-gaussian-test",
            "13",
            "--batch-size",
            "5",
            "--output-dir",
            str(output_dir),
        ]
    )

    expected = {
        "forest.joblib",
        "history.csv",
        "report.txt",
        "test_predictions.npz",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}

    saved = np.load(output_dir / "test_predictions.npz")
    forest = joblib.load(output_dir / "forest.joblib")
    np.testing.assert_allclose(
        forest.predict(saved["points"]),
        saved["prediction"],
    )
    np.testing.assert_allclose(
        forest.predict(saved["gaussian_points"]),
        saved["gaussian_prediction"],
    )

    sections = _read_sections(output_dir / "report.txt")
    assert {
        "[configuration]",
        "[target]",
        "[performance]",
        "[test_oracle_range]",
        "[pointwise_relative_error_quantiles]",
        "[gaussian_weighted_performance]",
        "[gaussian_weighted_oracle_range]",
        "[gaussian_weighted_pointwise_relative_error_quantiles]",
        "[gaussian_weighted_timing_seconds]",
        "[random_forest]",
        "[timing_seconds]",
    } == sections
    report = (output_dir / "report.txt").read_text()
    assert "training_queries: 40" in report
    assert "tree_count: 3" in report
    assert "q99.9:" in report

    with (output_dir / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["training_queries"] == "40"
