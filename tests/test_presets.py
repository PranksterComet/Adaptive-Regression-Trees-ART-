from dataclasses import FrozenInstanceError

import pytest

from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.presets import DEFAULT_TREE_PRESET, resolve_min_side_points


def test_default_tree_preset_uses_recommended_tree_settings() -> None:
    preset = DEFAULT_TREE_PRESET

    assert preset.leaf_degree == 1
    assert preset.leaf_include_bias
    assert preset.splitter_name == "soft_oblique"
    assert preset.relative_error_floor == 1e-12
    assert preset.ridge == 0.0
    assert preset.ridge_solver == "auto"
    assert preset.min_split_gain == 0.0
    assert preset.min_relative_split_gain == 1e-3
    assert preset.max_retries_on_failure == 3
    assert not preset.profile_build_timing
    assert preset.sample_count is None
    assert preset.sampling.isotropic_sampling
    assert preset.sampling.feasibility_tol == 1e-10
    assert preset.sampling.max_feasible_tries == 20_000
    assert preset.temperature.strategy == "fixed"
    assert preset.temperature.c == 1.0
    assert preset.splitter.min_side_points is None
    assert preset.hinge_splitter.mode == "both"
    assert preset.hinge_splitter.mu == 1.0
    assert preset.hinge_splitter.tol == 1e-6
    assert preset.hinge_splitter.init_scale == 1e-2


@pytest.mark.parametrize(
    ("model", "input_dimension", "expected"),
    [
        (AffineRidgeModel(), 4, 5),
        (PolynomialRidgeModel(degree=2), 3, 10),
    ],
)
def test_min_side_points_defaults_to_model_effective_dimension(
    model: object,
    input_dimension: int,
    expected: int,
) -> None:
    value, policy = resolve_min_side_points(None, model, input_dimension)

    assert value == expected
    assert policy == "effective_dimension"


def test_explicit_min_side_points_overrides_effective_dimension() -> None:
    value, policy = resolve_min_side_points(7, AffineRidgeModel(), 20)

    assert value == 7
    assert policy == "explicit"


def test_min_side_points_rejects_nonpositive_override() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        resolve_min_side_points(0, AffineRidgeModel(), 2)


def test_tree_preset_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_TREE_PRESET.ridge = 1e-8
