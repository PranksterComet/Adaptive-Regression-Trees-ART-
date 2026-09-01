from __future__ import annotations

import numpy as np
import pytest

from art.models import AffineRidgeModel, augment_features
from art.splitters import HingeAffineSplitter


def hinge_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(300, 2))
    theta1 = np.array([1.2, -0.4, 0.3])
    theta2 = np.array([0.2, 0.6, 0.2])
    design = augment_features(X)
    return X, np.maximum(design @ theta1, design @ theta2)


@pytest.mark.parametrize("solver", ("auto", "normal", "qr", "svd"))
def test_hinge_splitter_uses_shared_ridge_solvers(solver: str) -> None:
    X, y = hinge_data()
    splitter = HingeAffineSplitter(
        mode="max",
        ridge=1e-8,
        solver=solver,
        auto_rcond_threshold=1e-10,
        mu=1.0,
        max_iters=100,
        tol=1e-8,
        min_side_points=8,
        min_side_fraction=0.05,
        n_restarts=20,
        random_state=11,
    )

    result = splitter.split(X, y)

    assert result.loss < 1e-12
    assert result.left_model.solver == solver
    assert result.right_model.solver == solver
    assert len(result.metadata["left_cond_estimate_history"]) == result.n_iters
    assert len(result.metadata["right_cond_estimate_history"]) == result.n_iters
    assert result.metadata["parent_solve"]["solver_requested"] == solver


def test_hinge_auto_solver_records_iteration_solver_histories() -> None:
    X, y = hinge_data()
    result = HingeAffineSplitter(
        mode="max",
        ridge=1e-8,
        solver="auto",
        max_iters=100,
        tol=1e-8,
        min_side_points=8,
        n_restarts=20,
        random_state=11,
    ).split(X, y)

    metadata = result.metadata
    for side in ("left", "right"):
        condition_history = metadata[f"{side}_cond_estimate_history"]
        estimator_history = metadata[f"{side}_condition_estimator_history"]
        solver_history = metadata[f"{side}_solver_history"]

        assert len(condition_history) == result.n_iters
        assert len(estimator_history) == result.n_iters
        assert len(solver_history) == result.n_iters
        assert np.all(np.isfinite(condition_history))
        assert set(solver_history) <= {"cholesky", "qr", "svd"}


def test_hinge_auto_solver_can_fall_back_to_qr() -> None:
    rng = np.random.default_rng(33)
    x = rng.uniform(-1.0, 1.0, size=300)
    q = rng.uniform(-1.0, 1.0, size=300)
    X = np.column_stack([x, x + 1e-7 * rng.normal(size=300), q])
    theta1 = np.array([0.3, -0.3, 1.0, 0.2])
    theta2 = np.array([-0.2, 0.2, -0.5, -0.2])
    design = augment_features(X)
    y = np.maximum(design @ theta1, design @ theta2)

    result = HingeAffineSplitter(
        mode="max",
        ridge=0.0,
        solver="auto",
        max_iters=100,
        min_side_points=5,
        n_restarts=20,
        random_state=0,
    ).split(X, y)

    selected_solvers = {
        *result.metadata["left_solver_history"],
        *result.metadata["right_solver_history"],
    }
    assert "qr" in selected_solvers
    assert result.loss < 1e-20


def test_hinge_fixed_solver_omits_solver_histories() -> None:
    X, y = hinge_data()
    result = HingeAffineSplitter(
        mode="max",
        solver="qr",
        max_iters=20,
        min_side_points=8,
        n_restarts=20,
        random_state=11,
    ).split(X, y)

    assert "left_solver_history" not in result.metadata
    assert "right_solver_history" not in result.metadata


def test_hinge_initialization_balances_shifted_data() -> None:
    rng = np.random.default_rng(19)
    X = rng.uniform([20.0, -40.0], [24.0, -35.0], size=(151, 2))
    X_aug = augment_features(X)
    parent_theta = np.array([0.2, -0.1, 3.0])
    splitter = HingeAffineSplitter(init_scale=0.03)

    theta1, theta2 = splitter._initialize_thetas(
        X_aug,
        parent_theta,
        np.random.default_rng(5),
    )
    right = X_aug @ (theta1 - theta2) >= 0.0

    assert abs(int(np.sum(right)) - int(np.sum(~right))) <= 1
    np.testing.assert_allclose((theta1 + theta2) / 2.0, parent_theta)


def test_hinge_result_hard_refits_both_tree_sides() -> None:
    rng = np.random.default_rng(23)
    X = rng.uniform(-2.0, 2.0, size=(200, 2))
    y = X[:, 0] ** 2 + 0.4 * X[:, 0] * X[:, 1] - 0.2 * X[:, 1] ** 2
    parent_model = AffineRidgeModel(ridge=0.0, solver="qr").fit(X, y)
    parent_loss = np.mean((y - parent_model.predict(X)) ** 2)
    result = HingeAffineSplitter(
        mode="max",
        ridge=0.0,
        solver="qr",
        max_iters=0,
        min_side_points=5,
        n_restarts=1,
        random_state=7,
    ).split(X, y, parent_model=parent_model, parent_loss=parent_loss)

    right = X @ result.w - result.z >= 0.0
    expected_left = AffineRidgeModel(ridge=0.0, solver="qr").fit(
        X[~right],
        y[~right],
    )
    expected_right = AffineRidgeModel(ridge=0.0, solver="qr").fit(
        X[right],
        y[right],
    )

    np.testing.assert_allclose(result.left_model.coef_, expected_left.coef_)
    np.testing.assert_allclose(result.right_model.coef_, expected_right.coef_)
    assert result.loss <= parent_loss + 1e-12
    assert result.metadata["hard_left_solve"]["solver_used"] == "qr"
    assert result.metadata["hard_right_solve"]["solver_used"] == "qr"
