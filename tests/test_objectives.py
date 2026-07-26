from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from art.models import AffineRidgeModel
from art.objectives import SoftObliqueRidgeObjective
from art.optimizers import LineSearchResult
from art.splitters import (
    SoftObliqueSplitter,
    SplitNotFoundError,
    project_unit_w_gradient,
)


class CountingAffineModel(AffineRidgeModel):
    def __init__(self, counter: dict[str, int], ridge: float = 1e-8):
        super().__init__(ridge=ridge)
        self.counter = counter

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        weight_floor: float = 1e-12,
    ) -> "CountingAffineModel":
        self.counter["weighted_fits"] += 1
        super().fit_weighted(X, y, weights, weight_floor)
        return self

    def clone(self) -> "CountingAffineModel":
        return CountingAffineModel(self.counter, ridge=self.ridge)


def piecewise_affine_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, size=(100, 2))
    y = np.where(
        X[:, 0] >= 0.0,
        0.8 - 0.4 * X[:, 0] + 1.2 * X[:, 1],
        -0.5 + 1.1 * X[:, 0] - 0.3 * X[:, 1],
    )
    return X, y


def test_objective_caches_value_and_gradient_separately() -> None:
    X, y = piecewise_affine_data()
    counter = {"weighted_fits": 0}
    objective = SoftObliqueRidgeObjective(
        temperature=0.15,
        model_template=CountingAffineModel(counter),
    )
    theta = np.array([1.0, 0.0, 0.0])

    evaluation = objective.evaluate(theta, X, y)

    assert counter["weighted_fits"] == 2
    assert evaluation.loss is None
    assert evaluation.grad is None

    loss = objective.value(evaluation)
    gradient = objective.grad(evaluation)

    assert objective.value(evaluation) == loss
    assert objective.grad(evaluation) is gradient
    assert counter["weighted_fits"] == 2

    combined = objective.value_and_grad(theta, X, y)
    assert np.isclose(combined.loss, loss)
    assert np.allclose(combined.grad, gradient)
    assert counter["weighted_fits"] == 4

    assert np.isclose(objective.value(theta, X, y), loss)
    assert counter["weighted_fits"] == 6


def test_reference_evaluation_reuses_models_without_refitting() -> None:
    X, y = piecewise_affine_data()
    counter = {"weighted_fits": 0}
    objective = SoftObliqueRidgeObjective(
        temperature=0.15,
        model_template=CountingAffineModel(counter),
    )
    reference = objective.value_and_grad(np.array([1.0, 0.0, 0.0]), X, y)

    trial = objective.evaluate(
        np.array([0.98, 0.2, 0.05]),
        X,
        y,
        reference=reference,
    )

    assert counter["weighted_fits"] == 2
    assert trial.left_model is reference.left_model
    assert trial.right_model is reference.right_model
    assert trial.metadata["models_refit"] is False
    assert trial.loss is None
    assert trial.grad is None
    assert np.isfinite(objective.value(trial))
    assert np.all(np.isfinite(objective.grad(trial)))


def make_counting_splitter(
    counter: dict[str, int],
    refit_during_line_search: bool,
) -> SoftObliqueSplitter:
    return SoftObliqueSplitter(
        model_template=CountingAffineModel(counter),
        temperature=0.15,
        max_iters=1,
        grad_atol=0.0,
        grad_rtol=0.0,
        min_side_points=2,
        n_restarts=1,
        alpha0=0.1,
        adaptive_alpha=False,
        max_backtracks=10,
        refit_during_line_search=refit_during_line_search,
        random_state=7,
    )


def assert_final_gradient_matches_returned_split(
    result,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    theta = np.concatenate([result.w, np.array([result.z])])
    objective = SoftObliqueRidgeObjective(
        temperature=0.15,
        model_template=AffineRidgeModel(),
    )
    evaluation = objective.value_and_grad(theta, X, y)
    projected = project_unit_w_gradient(theta, evaluation.grad)
    expected = float(np.linalg.norm(projected))

    assert np.isclose(result.metadata["final_grad_norm"], expected)


def test_splitter_reuses_accepted_refitted_evaluation() -> None:
    X, y = piecewise_affine_data()
    counter = {"weighted_fits": 0}
    result = make_counting_splitter(counter, True).split(X, y)
    n_candidate_evaluations = result.metadata["backtrack_history"][0] + 1

    assert result.n_iters == 1
    assert counter["weighted_fits"] == 2 * (1 + n_candidate_evaluations)
    assert result.metadata["refit_during_line_search"] is True
    assert len(result.metadata["projected_grad_norm_history"]) == result.n_iters + 1
    assert (
        result.metadata["final_grad_norm"]
        == result.metadata["projected_grad_norm_history"][-1]
    )
    assert_final_gradient_matches_returned_split(result, X, y)


def test_frozen_line_search_refits_only_the_accepted_candidate() -> None:
    X, y = piecewise_affine_data()
    counter = {"weighted_fits": 0}
    result = make_counting_splitter(counter, False).split(X, y)

    assert result.n_iters == 1
    assert counter["weighted_fits"] == 4
    assert result.metadata["refit_during_line_search"] is False
    assert len(result.metadata["line_search_loss_history"]) == 1
    assert len(result.metadata["soft_loss_history"]) == 2
    assert len(result.metadata["projected_grad_norm_history"]) == result.n_iters + 1
    assert (
        result.metadata["final_grad_norm"]
        == result.metadata["projected_grad_norm_history"][-1]
    )
    assert_final_gradient_matches_returned_split(result, X, y)


def test_gradient_history_contains_initial_state_with_zero_iterations() -> None:
    X, y = piecewise_affine_data()
    splitter = SoftObliqueSplitter(
        model_template=AffineRidgeModel(),
        temperature=0.15,
        max_iters=0,
        min_side_points=2,
        random_state=7,
    )

    result = splitter.split(X, y)
    grad_history = result.metadata["projected_grad_norm_history"]

    assert result.n_iters == 0
    assert len(result.metadata["soft_loss_history"]) == 1
    assert len(grad_history) == 1
    assert result.metadata["initial_grad_norm"] == grad_history[0]
    assert result.metadata["final_grad_norm"] == grad_history[0]
    assert_final_gradient_matches_returned_split(result, X, y)


def test_accepted_step_below_alpha_min_does_not_stop(monkeypatch) -> None:
    X, y = piecewise_affine_data()
    accepted_step = 1e-13

    def accept_tiny_step(
        value_fn,
        candidate_fn,
        current_value,
        directional_derivative,
        **kwargs,
    ) -> LineSearchResult:
        del current_value, directional_derivative, kwargs
        theta = candidate_fn(accepted_step)
        return LineSearchResult(
            theta=theta,
            value=value_fn(theta),
            step_size=accepted_step,
            success=True,
            n_backtracks=0,
        )

    monkeypatch.setattr("art.splitters.armijo_backtracking", accept_tiny_step)
    splitter = SoftObliqueSplitter(
        model_template=AffineRidgeModel(),
        temperature=0.15,
        max_iters=2,
        grad_atol=0.0,
        grad_rtol=0.0,
        min_side_points=2,
        alpha0=1e-12,
        alpha_min=1e-12,
        alpha_grow=10.0,
        adaptive_alpha=True,
        random_state=7,
    )

    result = splitter.split(X, y)

    assert result.n_iters == 2
    assert result.stop_reason == "max_iters"
    assert result.metadata["step_size_history"] == [accepted_step, accepted_step]
    assert result.metadata["alpha_min_saturated"] is True


def test_internal_restarts_return_best_valid_result(monkeypatch) -> None:
    X, y = piecewise_affine_data()
    splitter = SoftObliqueSplitter(
        model_template=AffineRidgeModel(),
        temperature=0.15,
        n_restarts=3,
        random_state=7,
    )
    scripted = iter(
        [
            SimpleNamespace(loss=3.0, metadata={"run": 1}),
            SimpleNamespace(loss=1.0, metadata={"run": 2}),
            SimpleNamespace(loss=2.0, metadata={"run": 3}),
        ]
    )
    monkeypatch.setattr(
        splitter,
        "_optimize_from_initialization",
        lambda **kwargs: next(scripted),
    )

    result = splitter.split(X, y)

    assert result.loss == 1.0
    assert result.metadata["run"] == 2


def test_internal_restarts_report_last_failed_run(monkeypatch) -> None:
    X, y = piecewise_affine_data()
    splitter = SoftObliqueSplitter(
        model_template=AffineRidgeModel(),
        temperature=0.15,
        n_restarts=3,
        random_state=7,
    )
    calls = 0

    def fail_initialization(**kwargs):
        nonlocal calls
        calls += 1
        raise SplitNotFoundError(
            "min_side_points",
            "scripted failure",
            diagnostics={"run": calls},
        )

    monkeypatch.setattr(
        splitter,
        "_optimize_from_initialization",
        fail_initialization,
    )

    with pytest.raises(SplitNotFoundError) as exc_info:
        splitter.split(X, y)

    assert calls == 3
    assert exc_info.value.reason == "min_side_points"
    assert exc_info.value.diagnostics["run"] == 3
