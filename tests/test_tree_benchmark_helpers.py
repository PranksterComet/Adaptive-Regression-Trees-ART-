from __future__ import annotations

import numpy as np
import pytest

from art.domain import BoxDomain
from art.metrics import mean_squared_error, relative_l2_error
from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.splitters import HingeAffineSplitter, SoftObliqueSplitter
from art.tree import LeafNode, RegressionTree
from examples.tree_2D_benchmark import parse_args as parse_2d_args
from examples.tree_benchmark_helpers import (
    evaluate_tree_benchmark,
    make_benchmark_splitter,
    pointwise_test_errors,
)
from examples.tree_high_dim_benchmark import make_benchmark_target, parse_args


class ZeroModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


def test_pointwise_test_errors_aggregate_to_selected_metrics() -> None:
    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([0.5, 1.0, 5.0, 10.0])

    squared, squared_label = pointwise_test_errors("mse", y_true, y_pred, 1e-12)
    relative, relative_label = pointwise_test_errors(
        "relative_l2",
        y_true,
        y_pred,
        1e-12,
    )

    assert np.mean(squared) == mean_squared_error(y_true, y_pred)
    assert np.sqrt(np.mean(relative**2)) == relative_l2_error(y_true, y_pred)
    assert squared_label == "squared error"
    assert relative_label == "normalized relative L2 contribution"


def test_batched_test_evaluation_matches_single_batch() -> None:
    domain = BoxDomain.hypercube(3, -1.0, 1.0)
    tree = RegressionTree(LeafNode(ZeroModel()))

    def target(X: np.ndarray) -> np.ndarray:
        return np.sum(X, axis=1) + 2.0

    single = evaluate_tree_benchmark(
        tree,
        target,
        domain,
        n_test=23,
        error_metric="relative_l2",
        relative_error_floor=1e-12,
        random_state=9,
    )
    batched = evaluate_tree_benchmark(
        tree,
        target,
        domain,
        n_test=23,
        error_metric="relative_l2",
        relative_error_floor=1e-12,
        random_state=9,
        batch_size=5,
    )

    assert single.metrics == batched.metrics
    assert single.pointwise_relative_quantiles == batched.pointwise_relative_quantiles
    assert single.oracle_min == batched.oracle_min
    assert single.oracle_max == batched.oracle_max
    assert single.oracle_min <= single.oracle_max
    np.testing.assert_array_equal(
        single.selected_point_errors,
        batched.selected_point_errors,
    )


def test_high_dimensional_quadratic_uses_default_paper_spectrum() -> None:
    args = parse_args(["--benchmark", "quadratic", "--dimension", "5", "--seed", "3"])
    domain = BoxDomain.hypercube(5, -3.0, 3.0)

    target, metadata = make_benchmark_target(args, domain)

    np.testing.assert_allclose(
        np.diag(target.Lambda),
        np.arange(1, 6, dtype=float) ** -3,
    )
    np.testing.assert_allclose(target.Q.T @ target.Q, np.eye(5), atol=1e-12)
    assert metadata["rotation_mode"] == "random"


def test_high_dimensional_sphere_radius_matches_requested_fraction() -> None:
    args = parse_args(
        [
            "--benchmark",
            "spherical_piecewise",
            "--dimension",
            "4",
            "--sphere-volume-fraction",
            "0.35",
            "--sphere-volume-probe-size",
            "30000",
            "--seed",
            "5",
        ]
    )
    domain = BoxDomain.hypercube(4, -3.0, 3.0)

    target, metadata = make_benchmark_target(args, domain)
    rng = np.random.default_rng(17)
    points = rng.uniform(-3.0, 3.0, size=(30_000, 4))
    measured = np.mean(np.linalg.norm(points - target.center, axis=1) <= target.radius)

    assert abs(measured - 0.35) < 0.015
    assert metadata["radius_source"] == "estimated_volume_fraction"


def test_shared_benchmark_arguments_construct_hrt_splitter() -> None:
    args = parse_args(
        [
            "--splitter",
            "hrt",
            "--leaf-degree",
            "1",
            "--hrt-mode",
            "min",
            "--hrt-mu",
            "0.4",
            "--hrt-tol",
            "2e-7",
            "--hrt-init-scale",
            "0.03",
            "--max-iters",
            "17",
            "--min-side-fraction",
            "0.1",
            "--n-restarts",
            "1",
            "--ridge-solver",
            "qr",
        ]
    )
    model = AffineRidgeModel(ridge=args.ridge, solver=args.ridge_solver)

    splitter, temperature_config = make_benchmark_splitter(args, model, 6)

    assert isinstance(splitter, HingeAffineSplitter)
    assert splitter.mode == "min"
    assert splitter.mu == 0.4
    assert splitter.tol == 2e-7
    assert splitter.init_scale == 0.03
    assert splitter.max_iters == 17
    assert splitter.min_side_points == 6
    assert splitter.min_side_fraction == 0.1
    assert splitter.solver == "qr"
    assert temperature_config is None


def test_shared_benchmark_default_still_constructs_soft_splitter() -> None:
    args = parse_args(["--dimension", "2"])
    model = AffineRidgeModel(ridge=args.ridge, solver=args.ridge_solver)

    splitter, temperature_config = make_benchmark_splitter(args, model, 3)

    assert isinstance(splitter, SoftObliqueSplitter)
    assert temperature_config is not None
    assert temperature_config.strategy == args.temperature_strategy


def test_hrt_rejects_polynomial_leaves() -> None:
    args = parse_args(["--splitter", "hrt", "--leaf-degree", "2"])
    model = PolynomialRidgeModel(degree=2)

    with pytest.raises(ValueError, match="requires --leaf-degree 1"):
        make_benchmark_splitter(args, model, 6)


def test_2d_parser_exposes_hrt_parameters() -> None:
    args = parse_2d_args(
        [
            "--splitter",
            "hrt",
            "--hrt-mode",
            "max",
            "--hrt-mu",
            "0.25",
            "--hrt-tol",
            "1e-8",
            "--hrt-init-scale",
            "0.02",
            "--profile-build-timing",
        ]
    )

    assert args.splitter == "hrt"
    assert args.hrt_mode == "max"
    assert args.hrt_mu == 0.25
    assert args.hrt_tol == 1e-8
    assert args.hrt_init_scale == 0.02
    assert args.profile_build_timing
