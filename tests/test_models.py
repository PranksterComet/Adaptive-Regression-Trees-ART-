import numpy as np
import pytest

from art.models import (
    AffineRidgeModel,
    PolynomialRidgeModel,
    augment_features,
    solve_weighted_ridge,
)


def test_affine_initialization_preloads_scipy_linalg(monkeypatch) -> None:
    import builtins

    imported_modules = []
    original_import = builtins.__import__

    def track_import(name, *args, **kwargs):
        imported_modules.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    AffineRidgeModel()

    assert "scipy.linalg" in imported_modules


def test_affine_prediction_matches_augmented_matrix_product() -> None:
    model = AffineRidgeModel()
    model.coef_ = np.array([1.5, -2.0, 0.75])
    X = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]])

    expected = augment_features(X) @ model.coef_

    np.testing.assert_allclose(model.predict(X), expected)


def test_affine_prediction_accepts_one_sample() -> None:
    model = AffineRidgeModel()
    model.coef_ = np.array([2.0, -1.0, 0.5])

    prediction = model.predict(np.array([3.0, 4.0]))

    assert prediction.shape == (1,)
    np.testing.assert_allclose(prediction, np.array([2.5]))


@pytest.mark.parametrize("solver", ("auto", "normal", "qr", "svd"))
def test_polynomial_prepared_fit_matches_raw_fit(solver: str) -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 3))
    y = (
        0.5
        + X[:, 0] ** 2
        - 0.4 * X[:, 0] * X[:, 1]
        + 0.2 * X[:, 2]
    )
    template = PolynomialRidgeModel(degree=2, ridge=1e-8, solver=solver)
    prepared = template.prepare_design(X)

    raw_model = template.clone().fit(X, y)
    prepared_model = template.clone().fit_design(prepared, y)

    np.testing.assert_allclose(
        prepared_model.predict_design(prepared),
        raw_model.predict(X),
    )


@pytest.mark.parametrize("solver", ("auto", "normal", "qr", "svd"))
def test_polynomial_prepared_weighted_fit_matches_raw_fit(solver: str) -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(35, 2))
    y = 1.0 - X[:, 0] + 0.3 * X[:, 1] ** 2
    weights = rng.uniform(0.1, 1.0, size=X.shape[0])
    template = PolynomialRidgeModel(degree=3, ridge=1e-7, solver=solver)
    prepared = template.prepare_design(X)

    raw_model = template.clone().fit_weighted(X, y, weights)
    prepared_model = template.clone().fit_weighted_design(prepared, y, weights)

    np.testing.assert_allclose(
        prepared_model.predict_design(prepared),
        raw_model.predict(X),
    )


def test_weighted_ridge_solvers_agree_on_well_conditioned_design() -> None:
    rng = np.random.default_rng(14)
    design = rng.normal(size=(80, 7))
    y = rng.normal(size=80)
    weights = rng.uniform(0.05, 1.0, size=80)

    coefficients = {
        solver: solve_weighted_ridge(
            design, y, weights, ridge=1e-6, solver=solver
        ).coefficients
        for solver in ("auto", "normal", "qr", "svd")
    }

    np.testing.assert_allclose(coefficients["auto"], coefficients["normal"])
    np.testing.assert_allclose(coefficients["qr"], coefficients["normal"])
    np.testing.assert_allclose(coefficients["svd"], coefficients["normal"])


def test_auto_solver_uses_cholesky_for_well_conditioned_design() -> None:
    rng = np.random.default_rng(15)
    design = rng.normal(size=(100, 6))
    result = solve_weighted_ridge(
        design,
        rng.normal(size=100),
        np.ones(100),
        ridge=0.0,
        solver="auto",
    )

    assert result.solver_requested == "auto"
    assert result.solver_used == "cholesky"
    assert result.condition_estimator == "sqrt_gram_1norm"
    assert result.cond_estimate is not None
    assert result.cond_estimate < 1e5
    assert result.fallback_reason is None


def test_auto_solver_falls_back_to_qr_for_ill_conditioned_design() -> None:
    rng = np.random.default_rng(16)
    base = rng.normal(size=120)
    design = np.column_stack(
        [base, base + 1e-7 * rng.normal(size=120), rng.normal(size=120)]
    )
    result = solve_weighted_ridge(
        design,
        rng.normal(size=120),
        np.ones(120),
        ridge=0.0,
        solver="auto",
    )

    assert result.solver_used == "qr"
    assert result.cond_estimate is not None
    assert result.cond_estimate >= 1e5
    assert result.fallback_reason in ("rcond_below_threshold", "cholesky_failed")


def test_explicit_qr_omits_condition_estimate() -> None:
    design = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    result = solve_weighted_ridge(
        design,
        np.arange(8, dtype=float),
        np.ones(8),
        ridge=0.0,
        solver="qr",
    )

    assert result.solver_used == "qr"
    assert result.condition_estimator is None
    assert result.cond_estimate is None


def test_svd_reports_augmented_design_condition() -> None:
    design = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    result = solve_weighted_ridge(
        design,
        np.arange(8, dtype=float),
        np.ones(8),
        ridge=0.0,
        solver="svd",
    )

    assert result.condition_estimator == "svd_2norm"
    assert np.isclose(result.cond_estimate, np.linalg.cond(design))


def test_normal_solver_reports_cholesky_failure_for_singular_gram() -> None:
    design = np.column_stack([np.ones(8), np.zeros(8)])
    y = np.arange(8, dtype=float)

    with pytest.raises(np.linalg.LinAlgError, match="Cholesky factorization failed"):
        solve_weighted_ridge(
            design,
            y,
            np.ones_like(y),
            ridge=0.0,
            solver="normal",
        )


@pytest.mark.parametrize("model", (
    AffineRidgeModel(solver="auto", auto_rcond_threshold=1e-8),
    PolynomialRidgeModel(degree=2, solver="svd"),
))
def test_clone_preserves_ridge_solver(model) -> None:
    cloned = model.clone()
    assert cloned.solver == model.solver
    assert cloned.auto_rcond_threshold == model.auto_rcond_threshold


def test_invalid_ridge_solver_is_rejected() -> None:
    with pytest.raises(ValueError, match="solver must be one of"):
        PolynomialRidgeModel(solver="invalid")
