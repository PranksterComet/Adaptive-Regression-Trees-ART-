"""Defer child fits until top-up unless hard losses are needed by the builder."""

import numpy as np
import pytest

from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.splitters import SoftObliqueSplitter
from art.temperature import TemperatureConfig
from art.tree import SplitNode
from test_batched_builder import make_builder


@pytest.mark.parametrize("strategy,restarts,gain,enabled", [
    ("fixed", 1, False, False), ("splitter", 1, False, False),
    ("fixed", 2, False, True), ("fixed", 1, True, True),
    ("tune_root", 1, False, True), ("tune_node", 1, False, True),
])
def test_builder_announces_only_enabled_hard_refits(capsys, strategy, restarts, gain, enabled):
    s = SoftObliqueSplitter(AffineRidgeModel(), 1., n_restarts=restarts)
    b = make_builder(splitter=s, check_split_gain=gain,
                     temperature_config=TemperatureConfig(strategy=strategy))
    assert b.hard_refits == enabled
    assert capsys.readouterr().out == ("Final hard refits enabled.\n" if enabled else "")


@pytest.mark.parametrize("degree", [1, 2])
def test_cpu_split_can_return_only_geometry_without_fitting_partition_models(monkeypatch, degree):
    X = np.random.default_rng(910).uniform(-.5, .5, (20, 2))
    X[-1] = [2., 3.]  # Exactly one observation on the right, fewer than p.
    y = np.sin(X[:, 0]) + X[:, 1]**2
    model = (AffineRidgeModel(ridge=0., solver="auto") if degree == 1 else
             PolynomialRidgeModel(ridge=0., solver="auto"))
    parent = model.clone().fit(X, y)
    loss = np.mean((parent.predict(X) - y)**2)
    s = SoftObliqueSplitter(model, 1., max_iters=0)
    monkeypatch.setattr(s, "_initialize_theta", lambda X, rng: np.array([1., 0., 1.]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Final hard models must not be fitted")

    with monkeypatch.context() as patch:
        patch.setattr(type(model), "fit", forbidden)
        if degree == 2:
            patch.setattr(type(model), "fit_design", forbidden)
        result = s.split(X, y, parent_model=parent, parent_loss=loss, hard_refit=False)
    assert (result.n_left, result.n_right) == (19, 1)
    assert result.left_model is result.right_model is None
    assert result.loss is result.split_gain is result.relative_split_gain is None
    assert not result.has_hard_fit and not result.metadata["hard_refit"]
    with pytest.raises(RuntimeError, match="no hard models"):
        result.predict(X)
    # Standalone calls still return fully fitted models by default.
    fitted = s.split(X, y, parent_model=parent, parent_loss=loss)
    assert fitted.has_hard_fit and np.isfinite(fitted.predict(X)).all()


@pytest.mark.parametrize("traversal", ["dfs", "bfs"])
def test_cpu_builder_tops_up_small_partition_before_fitting(monkeypatch, traversal):
    model = PolynomialRidgeModel(ridge=0., solver="auto")
    s = SoftObliqueSplitter(model.clone(), 1., max_iters=0)

    def initial(X, rng):
        x = np.sort(X[:, 0])
        return np.array([1., 0., .5 * (x[-1] + x[-2])])

    monkeypatch.setattr(s, "_initialize_theta", initial)
    b = make_builder(
        splitter=s, model_template=model, max_depth=1, traversal=traversal,
        temperature_config=TemperatureConfig(strategy="splitter"),
        oracle=lambda X: X[:, 0]**4 + .2 * X[:, 1]**2,
    )
    calls, fit = [], b._fit_node_model

    def cpu_fit(X, *args):
        calls.append(len(X))
        return fit(X, *args)

    monkeypatch.setattr(b, "_fit_node_model", cpu_fit)
    result = b.build()
    root = result.tree.root
    assert isinstance(root, SplitNode) and root.right.metadata["n_inherited"] == 1
    assert calls == [20, 20, 20] and result.oracle_queries == 40
    assert root.metadata["split_mse"] is root.metadata["split_gain_mse"] is None
    assert root.metadata["hard_refit"] is False
    assert result.tree.metadata["hard_refits"] is False
    for leaf in result.tree.iter_leaves():
        ref = model.clone().fit(leaf.metadata["X"], leaf.metadata["y"])
        np.testing.assert_array_equal(leaf.model.coef_, ref.coef_)
    assert np.isfinite(result.tree.predict(np.zeros((3, 2)))).all()


def test_multiple_restarts_require_hard_losses():
    s = SoftObliqueSplitter(AffineRidgeModel(), 1., n_restarts=2)
    with pytest.raises(ValueError, match="requires n_restarts == 1"):
        s.split(np.ones((10, 2)), np.ones(10), hard_refit=False)
