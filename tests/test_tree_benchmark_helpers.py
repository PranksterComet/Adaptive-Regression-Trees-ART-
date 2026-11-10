from __future__ import annotations

import csv

import numpy as np
import pytest

from art.domain import BoxDomain
from art.metrics import mean_squared_error, relative_l2_error
from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.sampling import HitAndRunSampler, ParallelNumbaBatchedHitAndRunSampler
from art.splitters import HingeAffineSplitter, SoftObliqueSplitter
from art.tree import LeafNode, RegressionTree, SplitNode
from examples.tree_2D_benchmark import (
    make_benchmark_target as make_2d_benchmark_target,
    parse_args as parse_2d_args,
)
from examples.benchmark_functions import (
    FIXED_SPHERICAL_AFFINE_OFFSET,
    fixed_spherical_affine_radius,
)
from examples.tree_benchmark_helpers import (
    evaluate_tree_benchmark,
    make_benchmark_samplers,
    make_benchmark_splitter,
    build_tree_benchmark,
    pointwise_test_errors,
    write_node_diagnostics,
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
    assert single.bad_node_test_fractions == batched.bad_node_test_fractions


def test_uniform_evaluation_measures_bad_node_volume() -> None:
    domain = BoxDomain.hypercube(1, -1.0, 1.0)
    tree = RegressionTree(
        SplitNode(
            w=np.array([1.0]),
            z=0.0,
            left=SplitNode(
                w=np.array([1.0]),
                z=-0.5,
                left=LeafNode(ZeroModel(), status="min_side_points"),
                right=LeafNode(ZeroModel(), status="tolerance_met"),
                status="split_with_warnings",
                metadata={"warnings": ("small_init_grad",)},
            ),
            right=LeafNode(ZeroModel(), status="max_depth"),
            status="split",
        )
    )

    evaluation = evaluate_tree_benchmark(
        tree,
        lambda X: np.ones(X.shape[0]),
        domain,
        n_test=20_000,
        error_metric="mse",
        relative_error_floor=1e-12,
        random_state=5,
        batch_size=137,
    )

    fractions = evaluation.bad_node_test_fractions
    assert fractions is not None
    assert fractions["min_side_points"] == pytest.approx(0.25, abs=0.01)
    assert fractions["small_init_grad"] == pytest.approx(0.5, abs=0.01)
    assert fractions["max_depth"] == pytest.approx(0.5, abs=0.01)
    assert fractions["combined"] == pytest.approx(1.0, abs=0.01)


def test_tree_evaluation_accepts_a_custom_point_sampler() -> None:
    domain = BoxDomain.hypercube(2, -1.0, 1.0)
    tree = RegressionTree(LeafNode(ZeroModel()))

    def target(X: np.ndarray) -> np.ndarray:
        return X[:, 0] + 2.0

    def sampler(count: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        return np.column_stack((np.full(count, 0.5), np.zeros(count)))

    evaluation = evaluate_tree_benchmark(
        tree,
        target,
        domain,
        n_test=7,
        error_metric="mse",
        relative_error_floor=1e-12,
        random_state=9,
        batch_size=3,
        point_sampler=sampler,
    )

    assert evaluation.metrics["test_mse"] == 6.25
    assert evaluation.oracle_min == 2.5
    assert evaluation.oracle_max == 2.5
    assert evaluation.bad_node_test_fractions is None


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


def test_high_dimensional_gaussian_mixture_forces_fixed_preset() -> None:
    args = parse_args(
        [
            "--benchmark",
            "gaussian_mixture",
            "--dimension",
            "4",
            "--seed",
            "3",
            "--offset",
            "2.0",
        ]
    )
    domain = BoxDomain.hypercube(4, -1.0, 1.0)

    target, metadata = make_benchmark_target(args, domain)

    assert metadata["weights"] == [1.0, 1.0]
    assert not metadata["normalize_weights"]
    assert metadata["component_sharpness"] == [8.0, 8.0]
    assert metadata["component_condition_numbers"] == [16.0, 16.0]
    assert metadata["offset"] == 2.0
    assert target(np.zeros(4)) > 2.0


def test_high_dimensional_plane_wave_accepts_explicit_normal() -> None:
    args = parse_args(
        [
            "--benchmark",
            "plane_wave",
            "--dimension",
            "3",
            "--plane-wave-normal",
            "1",
            "-1",
            "1",
            "--plane-wave-phase",
            "0",
        ]
    )
    domain = BoxDomain.hypercube(3, -3.0, 3.0)

    target, metadata = make_benchmark_target(args, domain)

    np.testing.assert_allclose(target.normal, np.array([1.0, -1.0, 1.0]) / np.sqrt(3))
    assert target.phase == 0.0
    assert target.feature_mode == "explicit"
    assert metadata["feature_mode"] == "explicit"


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
    center = np.asarray(metadata["center"])
    measured = np.mean(
        np.linalg.norm(points - center, axis=1) <= metadata["radius"]
    )

    assert abs(measured - 0.35) < 0.015
    assert metadata["radius_source"] == "estimated_volume_fraction"


def test_high_dimensional_sphere_uses_fixed_tacc_preset_by_default() -> None:
    args = parse_args(
        [
            "--benchmark",
            "spherical_piecewise",
            "--dimension",
            "4",
        ]
    )
    domain = BoxDomain.hypercube(4, -3.0, 3.0)

    target, metadata = make_benchmark_target(args, domain)

    assert args.sphere_piece_mode == "fixed_affine"
    assert metadata["radius"] == fixed_spherical_affine_radius(4)
    assert metadata["radius_source"] == "fixed_50pct_preset"
    assert metadata["requested_volume_fraction"] == 0.5
    assert metadata["piece_mode"] == "fixed_affine"
    assert metadata["offset"] == FIXED_SPHERICAL_AFFINE_OFFSET
    assert target(np.zeros(4)) == pytest.approx(5.0 + 4.15)


def test_sphere_radius_and_offset_override_the_fixed_defaults() -> None:
    args = parse_args(
        [
            "--benchmark",
            "spherical_piecewise",
            "--dimension",
            "3",
            "--sphere-radius",
            "1.5",
            "--offset",
            "-0.5",
        ]
    )
    domain = BoxDomain.hypercube(3, -3.0, 3.0)

    _, metadata = make_benchmark_target(args, domain)

    assert metadata["radius"] == 1.5
    assert metadata["radius_source"] == "explicit"
    assert metadata["requested_volume_fraction"] is None
    assert metadata["offset"] == -0.5


def test_2d_sphere_uses_same_fixed_preset_and_default_offset() -> None:
    args = parse_2d_args(
        [
            "--benchmark",
            "spherical_piecewise",
        ]
    )
    domain = BoxDomain.hypercube(2, -3.0, 3.0)

    target, metadata = make_2d_benchmark_target(args, domain)

    assert args.sphere_radius is None
    assert args.sphere_volume_fraction is None
    assert args.sphere_piece_mode == "fixed_affine"
    assert args.offset is None
    assert metadata["radius"] == fixed_spherical_affine_radius(2)
    assert metadata["radius_source"] == "fixed_50pct_preset"
    assert metadata["offset"] == FIXED_SPHERICAL_AFFINE_OFFSET
    assert target(np.zeros(2)) == pytest.approx(5.0 + 4.15)


@pytest.mark.parametrize("dimension", (2, 4))
def test_random_sphere_pieces_default_to_zero_offset(dimension: int) -> None:
    if dimension == 2:
        args = parse_2d_args(
            ["--benchmark", "spherical_piecewise", "--sphere-piece-mode", "random"]
        )
        domain = BoxDomain.hypercube(2, -3.0, 3.0)
        target, metadata = make_2d_benchmark_target(args, domain)
    else:
        args = parse_args(
            [
                "--benchmark",
                "spherical_piecewise",
                "--dimension",
                str(dimension),
                "--sphere-piece-mode",
                "random",
            ]
        )
        domain = BoxDomain.hypercube(dimension, -3.0, 3.0)
        target, metadata = make_benchmark_target(args, domain)

    assert args.offset is None
    assert metadata["offset"] == 0.0
    assert metadata["requested_volume_fraction"] == 0.5
    assert np.isfinite(target(np.zeros(dimension)))


def test_explicit_2d_sphere_offset_replaces_fixed_default() -> None:
    args = parse_2d_args(
        ["--benchmark", "spherical_piecewise", "--offset", "0.75"]
    )
    domain = BoxDomain.hypercube(2, -3.0, 3.0)

    target, metadata = make_2d_benchmark_target(args, domain)

    assert metadata["offset"] == 0.75
    assert target(np.zeros(2)) == pytest.approx(5.0 + 0.75)


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
            "--min-side-fallback",
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
    assert splitter.min_side_fallback is True
    assert splitter.solver == "qr"
    assert temperature_config is None


def test_shared_benchmark_default_still_constructs_soft_splitter() -> None:
    args = parse_args(["--dimension", "2"])
    model = AffineRidgeModel(ridge=args.ridge, solver=args.ridge_solver)

    splitter, temperature_config = make_benchmark_splitter(args, model, 3)

    assert isinstance(splitter, SoftObliqueSplitter)
    assert splitter.min_side_fallback is True
    assert temperature_config is not None
    assert temperature_config.strategy == args.temperature_strategy
    assert args.store_leaf_representative is False
    assert args.isotropic_sampling is False
    assert args.min_side_points == 1
    assert args.min_side_fraction == .05
    assert not args.check_split_gain
    assert args.traversal == "dfs" and args.optimizer == "cpu"


@pytest.mark.parametrize("optimizer", ["cpu", "hybrid_gram", "cupy_qr", "cupy_cholesky",
                                      "cupy_batched_cholesky"])
def test_shared_parser_forwards_gpu_options_without_loading_cuda(optimizer):
    args = parse_args(["--dimension", "2", "--optimizer", optimizer,
                       "--traversal", "bfs", "--node-batch-size", "7", "--no-batch-reorder",
                       "--backtrack-batch-size", "5", "--xw-refresh", "11", "--no-qr-parallel",
                       "--check-split-gain"])
    splitter, _ = make_benchmark_splitter(args, AffineRidgeModel(), args.min_side_points)
    assert splitter.optimizer == optimizer
    assert splitter.backtrack_batch_size == 5 and splitter.Xw_refresh == 11
    assert not splitter.qr_parallel
    assert args.traversal == "bfs" and args.node_batch_size == 7 and not args.batch_reorder
    assert args.check_split_gain


@pytest.mark.parametrize("check", [False, True])
def test_benchmark_builder_uses_explicit_side_counts_and_traversal(check):
    from art.domain import BoxDomain

    args = parse_args(["--dimension", "2", "--traversal", "bfs", "--node-batch-size", "7",
                       "--no-batch-reorder", "--leaf-degree", "2", "--sample-count", "20",
                       "--max-depth", "0", "--no-isotropic-sampling", "--no-check-split-gain"]
                      + (["--check-split-gain"] if check else []))
    run = build_tree_benchmark(args, BoxDomain.hypercube(2, -1., 1.),
                               lambda X: X[:, 0]**2 + X[:, 1])
    assert run.min_side_points == run.builder.splitter.min_side_points == 1
    assert run.builder.traversal == "bfs" and run.builder.node_batch_size == 7
    assert not run.builder.batch_reorder
    assert run.builder.check_split_gain == check
    assert run.tree.metadata["optimizer"] == "cpu"


def test_shared_benchmark_parser_enables_leaf_representatives() -> None:
    args = parse_args(["--dimension", "2", "--store-leaf-representative"])

    assert args.store_leaf_representative is True


def test_shared_benchmark_parser_exposes_validation_fraction() -> None:
    args = parse_args(["--dimension", "2", "--validate", "0.2"])

    assert args.validation_fraction == 0.2


def test_shared_benchmark_parser_disables_min_side_fallback() -> None:
    args = parse_args(["--dimension", "2", "--no-min-side-fallback"])
    model = AffineRidgeModel(ridge=args.ridge, solver=args.ridge_solver)

    splitter, _ = make_benchmark_splitter(args, model, 3)

    assert args.min_side_fallback is False
    assert splitter.min_side_fallback is False


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


def test_2d_parser_inherits_shared_tree_options() -> None:
    args = parse_2d_args(
        [
            "--min-side-fallback",
            "--store-leaf-representative",
            "--sampler",
            "parallel_numba",
        ]
    )

    assert args.min_side_fallback is True
    assert args.store_leaf_representative is True
    assert args.sampler == "parallel_numba"


def test_shared_parallel_sampler_cli_builds_requested_sampler() -> None:
    from numba import get_num_threads, set_num_threads

    args = parse_args(
        [
            "--sampler",
            "parallel_numba",
            "--n-chains",
            "17",
            "--sampler-threads",
            "3",
            "--no-parallel-random",
        ]
    )
    original_threads = get_num_threads()
    try:
        sampler, root_sampler = make_benchmark_samplers(args)
    finally:
        set_num_threads(original_threads)

    for configured in (sampler, root_sampler):
        assert isinstance(configured, ParallelNumbaBatchedHitAndRunSampler)
        assert configured.n_chains == 17
        assert configured.random_threads == 3
        assert not configured.parallel_random


def test_shared_sampler_cli_preserves_standard_sampler() -> None:
    args = parse_args(["--sampler", "hit_and_run"])
    sampler, root_sampler = make_benchmark_samplers(args)

    assert type(sampler) is HitAndRunSampler
    assert type(root_sampler) is HitAndRunSampler


def test_node_diagnostics_accepts_extra_columns(tmp_path) -> None:
    tree = RegressionTree(LeafNode(ZeroModel(), node_id="root"))
    path = tmp_path / "nodes.csv"

    write_node_diagnostics(
        tree,
        path,
        extra_columns=lambda node: {"plot_label": f"label:{node.node_id}"},
    )

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["node_id"] == "root"
    assert rows[0]["plot_label"] == "label:root"
