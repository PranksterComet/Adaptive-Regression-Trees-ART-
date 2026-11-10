from dataclasses import FrozenInstanceError

import pytest

from art.presets import BFS_GPU_TREE_PRESET, DEFAULT_TREE_PRESET


def test_default_tree_preset_uses_recommended_tree_settings() -> None:
    preset = DEFAULT_TREE_PRESET

    assert preset.leaf_degree == 1
    assert preset.leaf_include_bias
    assert preset.splitter_name == "soft_oblique"
    assert preset.relative_error_floor == 1e-12
    assert preset.ridge == 0.0
    assert preset.ridge_solver == "auto"
    assert preset.min_split_gain == 0.0
    assert preset.min_relative_split_gain == 0.0
    assert not preset.check_split_gain
    assert preset.traversal == "dfs"
    assert preset.node_batch_size == 20
    assert preset.batch_reorder
    assert preset.max_retries_on_failure == 3
    assert preset.min_side_fallback
    assert preset.validation_fraction == 0.0
    assert not preset.profile_build_timing
    assert preset.sample_count is None
    assert not preset.sampling.isotropic_sampling
    assert preset.sampling.sampler_name == "hit_and_run"
    assert preset.sampling.n_chains == 50
    assert preset.sampling.threads == 10
    assert preset.sampling.parallel_random
    assert preset.sampling.feasibility_tol == 1e-10
    assert preset.sampling.max_feasible_tries == 20_000
    assert preset.temperature.strategy == "fixed"
    assert preset.temperature.c == 1.0
    assert preset.splitter.min_side_points == 1
    assert preset.splitter.min_side_fraction == .05
    assert preset.splitter.optimizer == "cpu"
    assert preset.splitter.backtrack_batch_size == 8
    assert preset.splitter.Xw_refresh == 32
    assert preset.splitter.qr_parallel
    assert preset.hinge_splitter.mode == "both"
    assert preset.hinge_splitter.mu == 1.0
    assert preset.hinge_splitter.tol == 1e-6
    assert preset.hinge_splitter.init_scale == 1e-2


def test_bfs_gpu_preset_changes_only_traversal_and_optimizer():
    from dataclasses import replace

    assert BFS_GPU_TREE_PRESET == replace(
        DEFAULT_TREE_PRESET, traversal="bfs",
        splitter=replace(DEFAULT_TREE_PRESET.splitter, optimizer="cupy_batched_cholesky"),
    )


def test_tree_preset_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_TREE_PRESET.ridge = 1e-8
