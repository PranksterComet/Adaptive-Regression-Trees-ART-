"""BFS GPU dispatch policy, plus numerical builds on host-emulated/real CUDA."""

from dataclasses import replace

import numpy as np
import pytest

from cuda_cpu_backend import cuda_backend  # noqa: F401
from art import _cuda_batched_builder as dispatch
from art._cuda_batched_optimizer import BatchedOptimizerWorkspace
from art.metrics import max_pointwise_relative_error
from art.models import AffineRidgeModel, PolynomialRidgeModel, solve_weighted_ridge
from art.splitters import SoftObliqueSplitter, SplitNotFoundError
from art.temperature import TemperatureConfig
from art.tree import LeafNode, SplitNode
from test_batched_builder import make_builder
from test_builder import FixedAxisSplitter


def gpu_builder(**kwargs):
    model = kwargs.pop("model_template", AffineRidgeModel(ridge=1e-5))
    splitter = kwargs.pop("splitter", SoftObliqueSplitter(
        model.clone(), temperature=.3, optimizer="cupy_batched_cholesky",
        max_iters=2, min_side_points=3, min_side_fallback=True,
    ))
    options = dict(traversal="bfs", node_batch_size=3, model_template=model,
                   splitter=splitter, store_diagnostics=True, check_split_gain=True,
                   temperature_config=TemperatureConfig(strategy="splitter"))
    options.update(kwargs)
    return make_builder(**options)


@pytest.fixture
def scripted_workspace(monkeypatch):
    """Isolate queue/policy checks from CUDA and stochastic optimization."""
    made = []

    class Workspace:
        def __init__(self, s, capacity, n, d, **kwargs):
            self.s, self.capacity, self.reorder = s, capacity, kwargs["reorder"]
            self.host_X, self.host_y = np.empty((capacity, n, d)), np.empty((capacity, n))
            self.loads, self.calls, self.attempts = [], [], {}
            self.closed, self.script = False, None
            made.append(self)

        def load(self, X, y, *, temperature):
            assert np.shares_memory(X, self.host_X) and np.shares_memory(y, self.host_y)
            self.k = len(X)
            self.loads.append((X.copy(), y.copy(), temperature.copy()))
            self.attempts.clear()

        def optimize(self, *, nodes, parent_loss, min_side_fallback, hard_refit=True):
            self.calls.append((len(self.loads) - 1, tuple(nodes), min_side_fallback))
            results = []
            for j, loss in zip(nodes, parent_loss):
                attempt = self.attempts[j] = self.attempts.get(j, 0) + 1
                result = FixedAxisSplitter(median=True).split(
                    self.host_X[j], self.host_y[j], parent_loss=loss,
                )
                result.metadata["run"] = attempt
                if self.script is not None:
                    result = self.script(self, j, attempt, result)
                results.append(result)
            return results

        def close(self):
            self.closed = True

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", Workspace)
    return made


def test_one_workspace_packed_cohorts_and_final_cpu_leaves(scripted_workspace, monkeypatch):
    b = gpu_builder(batch_reorder=False, profile_build_timing=True)
    fits = []
    fit = b._fit_node_model

    def cpu_fit(*args):
        fits.append(len(args[0]))
        return fit(*args)

    monkeypatch.setattr(b, "_fit_node_model", cpu_fit)
    monkeypatch.setattr(b, "_split_node", lambda job: pytest.fail("scalar dispatch used"))
    result = b.build()
    ws, = scripted_workspace
    assert ws.closed and not ws.reorder
    assert [len(X) for X, _, _ in ws.loads] == [1, 2, 3, 1]
    assert len(fits) == result.tree.num_nodes() == 15
    assert result.oracle_queries == 160
    for X, y, temp in ws.loads:
        np.testing.assert_allclose(y, X[:, :, 0]**2 + .2 * X[:, :, 1]**2)
        np.testing.assert_array_equal(temp, np.full(len(X), .3))
    for leaf in result.tree.iter_leaves():
        np.testing.assert_allclose(leaf.model.predict(leaf.metadata["X"]),
                                   AffineRidgeModel(1e-5).fit(leaf.metadata["X"], leaf.metadata["y"])
                                   .predict(leaf.metadata["X"]))
    timing = result.build_timing
    assert 0 < timing["splitter_seconds"] <= timing["total_build_seconds"]


def test_selected_retries_reuse_load_and_enable_only_final_fallback(scripted_workspace, monkeypatch):
    b = gpu_builder(max_depth=2, max_retries_on_failure=2)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)

        def script(ws, j, attempt, result):
            if len(ws.loads) == 2:
                if j == 0 and attempt == 1:
                    return SplitNotFoundError("min_side_points", "scripted", diagnostics={"run": attempt})
                if j == 1 and attempt <= 2:
                    return replace(result, loss=result.parent_loss, split_gain=0., relative_split_gain=0.)
            return result

        ws.script = script
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    ws, = scripted_workspace
    assert ws.calls == [(0, (0,), False), (1, (0, 1), False),
                        (1, (0, 1), False), (1, (1,), True)]
    assert len(ws.loads) == 2 and result.restarts_on_failure == 3
    left, right = result.tree.root.left, result.tree.root.right
    assert left.metadata["split_attempt_failure_reasons"] == ("min_side_points",)
    assert right.metadata["split_attempt_failure_reasons"] == ("insufficient_split_gain",) * 2
    assert left.metadata["restarts_on_failure"] == 1
    assert right.metadata["restarts_on_failure"] == 2
    assert right.metadata["splitter_metadata"]["run"] == 3


def test_mixed_tuning_limits_do_not_repeat_a_slot_in_the_same_round(scripted_workspace, monkeypatch):
    b = gpu_builder(max_depth=2, max_retries_on_failure=1)
    resolved = []

    def temperature(X, y, depth, prepared):
        idx = len(resolved)
        temp, tuned = .2 + .1 * idx, idx == 1
        resolved.append(temp)
        return temp, {"temperature": temp, "temperature_tuned_at_node": tuned}, tuned

    monkeypatch.setattr(b, "_resolve_split_temperature", temperature)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)

        def script(ws, j, attempt, result):
            if len(ws.loads) == 2 and attempt == 1:
                return SplitNotFoundError("min_side_points", "scripted", diagnostics={"run": attempt})
            return result

        ws.script = script
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    ws, = scripted_workspace
    assert ws.calls == [(0, (0,), False), (1, (1,), False),
                        (1, (0,), True), (1, (1,), True)]
    np.testing.assert_allclose(ws.loads[1][2], [.3, .4])
    left = result.tree.root.left
    assert left.status == "min_side_points"
    assert left.metadata["temperature_tuned_at_node"] is True
    assert left.metadata["restarts_on_failure"] == 0
    assert left.metadata["splitter_metadata"] == {"run": 1}
    assert result.restarts_on_failure == 1


def test_temperature_failure_skips_slot_and_preserves_attachment(scripted_workspace, monkeypatch):
    b = gpu_builder(max_depth=2)
    resolve = b._resolve_split_temperature
    count = 0

    def temperature(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise SplitNotFoundError("temperature_tuning_failed", "scripted")
        return resolve(*args)

    monkeypatch.setattr(b, "_resolve_split_temperature", temperature)
    result = b.build()
    ws, = scripted_workspace
    assert [len(X) for X, _, _ in ws.loads] == [1, 1]
    assert result.tree.root.left.status == "temperature_tuning_failed"
    assert isinstance(result.tree.root.right, SplitNode)
    assert ws.closed


def test_cleanup_on_optimizer_exception(scripted_workspace, monkeypatch):
    b = gpu_builder()
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        ws.script = lambda *args: (_ for _ in ()).throw(RuntimeError("scripted GPU failure"))
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    with pytest.raises(RuntimeError, match="scripted GPU failure"):
        b.build()
    assert scripted_workspace[0].closed


def test_validation_failure_refits_cpu_leaf_on_all_rows(scripted_workspace, monkeypatch):
    b = gpu_builder(validation_fraction=.2, error_metric=max_pointwise_relative_error,
                    max_depth=1)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        ws.script = lambda *args: SplitNotFoundError("cholesky_failure", "scripted")
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    root = b.build().tree.root
    assert root.status == "cholesky_failure"
    assert root.metadata["convergence_fit_samples"] == 16
    ref = b.model_template.clone().fit(root.metadata["X"], root.metadata["y"])
    np.testing.assert_array_equal(root.model.coef_, ref.coef_)


@pytest.mark.parametrize("reason,retries", [("min_side_points", 2),
                                           ("insufficient_split_gain", 2),
                                           ("cholesky_failure", 0)])
def test_exhausted_attempts_keep_last_diagnostics(scripted_workspace, monkeypatch, reason, retries):
    b = gpu_builder(max_retries_on_failure=2)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)

        def script(ws, j, attempt, result):
            if reason == "insufficient_split_gain":
                return replace(result, loss=result.parent_loss, split_gain=0., relative_split_gain=0.)
            return SplitNotFoundError(reason, "scripted", diagnostics={"run": attempt})

        ws.script = script
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    root = result.tree.root
    ws, = scripted_workspace
    assert isinstance(root, LeafNode) and root.status == reason
    assert len(ws.loads) == 1 and len(ws.calls) == retries + 1
    assert result.restarts_on_failure == root.metadata["restarts_on_failure"] == retries
    assert root.metadata["split_attempt_failure_reasons"] == (reason,) * (retries + 1)
    assert root.metadata["splitter_metadata"]["run"] == retries + 1
    assert root.metadata["temperature"] == .3


def test_temperature_candidates_remain_scalar_and_tuned_node_has_no_retries(scripted_workspace,
                                                                           monkeypatch):
    b = gpu_builder(max_depth=1, max_retries_on_failure=2,
                    temperature_config=TemperatureConfig(strategy="tune_root", c_values=(.1, .3),
                                                         max_points=None))
    candidates = []

    def scalar(X, y, parent_model, parent_loss, temp, prepared, **kwargs):
        candidates.append((temp, kwargs["random_seed"]))
        return FixedAxisSplitter(median=True).split(X, y, parent_model, parent_loss, prepared)

    monkeypatch.setattr(b, "_run_splitter", scalar)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        ws.script = lambda *args: SplitNotFoundError("min_side_points", "scripted")
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    ws, = scripted_workspace
    assert len(candidates) == 2 and candidates[0][1] == candidates[1][1]
    assert candidates[0][0] < candidates[1][0]
    assert ws.calls == [(0, (0,), True)]
    assert result.restarts_on_failure == 0
    assert result.tree.root.metadata["temperature_tuned_at_node"] is True
    assert len(result.tree.root.metadata["temperature_candidate_scores"]) == 2


@pytest.mark.parametrize("traversal,optimizer", [(t, opt) for t in ("dfs", "bfs")
                                                for opt in ("cpu", "hybrid_gram", "cupy_qr", "cupy_cholesky")])
def test_other_configurations_keep_scalar_dispatch(monkeypatch, traversal, optimizer):
    s = SoftObliqueSplitter(AffineRidgeModel(), .3, optimizer=optimizer)
    b = gpu_builder(splitter=s, traversal=traversal, max_depth=1)
    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate",
                        lambda *a, **kw: pytest.fail("batched workspace allocated"))
    calls = []

    def scalar(job):
        calls.append(job.node_id)
        result = FixedAxisSplitter(median=True).split(job.samples.X, job.samples.y,
                                                     parent_loss=job.parent_mse)
        return b._finish_node(job, result)

    monkeypatch.setattr(b, "_split_node", scalar)
    assert isinstance(b.build().tree.root, SplitNode)
    assert calls == ["root"]


@pytest.mark.parametrize("degree,reorder", [(1, False), (1, True), (2, False), (2, True)])
def test_numerical_gpu_build_reuses_capacity_and_matches_numpy_fits(cuda_backend, monkeypatch,
                                                                  degree, reorder):
    model = AffineRidgeModel(ridge=1e-4) if degree == 1 else PolynomialRidgeModel(ridge=1e-4)
    b = gpu_builder(model_template=model, batch_reorder=reorder, sample_count=32,
                    max_depth=2, node_batch_size=3,
                    temperature_config=TemperatureConfig(strategy="fixed", c=.7),
                    oracle=lambda X: np.where(X[:, 0] >= .1, 1. + .3 * X[:, 1],
                                             -.5 + .2 * X[:, 1]**2))
    made, loads, exported = [], [], []
    allocate = BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        made.append(ws)
        load, optimize = ws.load, ws.optimize
        ptrs = ws.x.data.ptr, ws.ls.phi.data.ptr, ws.ls.y.data.ptr

        def load_cohort(X, y, **kw):
            assert np.shares_memory(X, ws.host_X) and np.shares_memory(y, ws.host_y)
            assert ptrs == (ws.x.data.ptr, ws.ls.phi.data.ptr, ws.ls.y.data.ptr)
            loads.append(len(X))
            return load(X, y, **kw)

        def optimize_cohort(*args, **kw):
            assert not args and "theta0" not in kw
            assert "parent_loss" in kw
            results = optimize(**kw)
            for j, result in zip(kw["nodes"], results):
                if isinstance(result, SplitNotFoundError):
                    continue
                X, y = ws.X[j], ws.y[j]
                phi = (np.column_stack((X, np.ones(len(X)))) if degree == 1 else
                       model.prepare_design(X).matrix)
                assert result.metadata["temperature"] == ws.temperature[j]
                mask = X @ result.w >= result.z
                for side, q in enumerate((~mask, mask)):
                    ref = solve_weighted_ridge(phi, y, q.astype(float), model.ridge,
                                               weight_floor=0., solver="normal")
                    fitted = result.left_model if side == 0 else result.right_model
                    np.testing.assert_allclose(fitted.coef_, ref.coefficients, rtol=2e-8, atol=2e-10)
                exported.append((result, X.copy(), result.predict(X).copy()))
            return results

        monkeypatch.setattr(ws, "load", load_cohort)
        monkeypatch.setattr(ws, "optimize", optimize_cohort)
        return ws

    monkeypatch.setattr(BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    ws, = made
    assert ws.blas_handle is None and ws.rng_handle is None
    assert ws.reorder == reorder and len(loads) >= 2 and max(loads) <= 3
    assert loads[0] == 1 and isinstance(result.tree.root, SplitNode)
    assert exported
    for split, X, predictions in exported:
        np.testing.assert_array_equal(split.predict(X), predictions)  # Survives load/close.
        assert split.metadata["initialization"] == "gpu"
    assert result.oracle_queries == sum(n.metadata["n_new"] for n in result.tree.iter_nodes())
    for node in result.tree.iter_nodes():
        assert node.status != "pending"
        assert node.region.contains(node.metadata["X"]).all()
        if isinstance(node, SplitNode):
            assert node.metadata["splitter_metadata"]["optimizer"] == "cupy_batched_cholesky"
        else:
            ref = model.clone().fit(node.metadata["X"], node.metadata["y"])
            np.testing.assert_array_equal(node.model.coef_, ref.coef_)
    assert np.isfinite(result.tree.predict(np.zeros((5, 2)))).all()


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_batch_reorder_requires_boolean(value):
    with pytest.raises(ValueError, match="batch_reorder"):
        gpu_builder(batch_reorder=value)


def test_batched_optimizer_requires_bfs_before_loading_cuda():
    with pytest.raises(ValueError, match="requires traversal='bfs'"):
        gpu_builder(traversal="dfs")


def test_batched_gain_checks_can_be_disabled(scripted_workspace, monkeypatch):
    b = gpu_builder(check_split_gain=False, max_depth=1, max_retries_on_failure=2,
                    min_split_gain=10., min_relative_split_gain=10.)
    allocate = dispatch.BatchedOptimizerWorkspace.allocate

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        ws.script = lambda ws, j, attempt, result: replace(
            result, loss=result.parent_loss + 1., split_gain=-1., relative_split_gain=-1.,
        )
        return ws

    monkeypatch.setattr(dispatch.BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    assert isinstance(result.tree.root, SplitNode)
    assert result.restarts_on_failure == 0
    assert len(scripted_workspace[0].calls) == 1


def test_gpu_builder_defers_small_partition_fit_until_topup(cuda_backend, monkeypatch):
    model = PolynomialRidgeModel(ridge=0., solver="auto")
    s = SoftObliqueSplitter(model.clone(), 1., optimizer="cupy_batched_cholesky", max_iters=0)
    b = gpu_builder(splitter=s, model_template=model, check_split_gain=False,
                    max_depth=1, sample_count=20,
                    oracle=lambda X: X[:, 0]**4 + .2 * X[:, 1]**2)
    allocate = BatchedOptimizerWorkspace.allocate
    proposals = []

    def setup(*args, **kwargs):
        ws = allocate(*args, **kwargs)
        optimize, fit = ws.optimize, ws._fit

        def soft_fit(**kw):
            assert not kw.get("hard", False)
            return fit(**kw)

        def controlled(**kw):
            assert kw["hard_refit"] is False
            theta = []
            for j in kw["nodes"]:
                x = np.sort(ws.X[j, :, 0])
                theta.append([1., 0., .5 * (x[-1] + x[-2])])
            results = optimize(np.array(theta), **kw)
            proposals.extend(results)
            return results

        monkeypatch.setattr(ws, "_fit", soft_fit)
        monkeypatch.setattr(ws, "optimize", controlled)
        return ws

    monkeypatch.setattr(BatchedOptimizerWorkspace, "allocate", setup)
    result = b.build()
    root = result.tree.root
    assert isinstance(root, SplitNode)
    assert root.right.metadata["n_inherited"] == 1
    assert root.metadata["hard_refit"] is False
    assert root.metadata["split_mse"] is None
    assert result.oracle_queries == 40 and result.tree.num_nodes() == 3
    assert proposals[0].right_model is None
    for leaf in result.tree.iter_leaves():
        assert leaf.metadata["n_samples"] == 20
        ref = model.clone().fit(leaf.metadata["X"], leaf.metadata["y"])
        np.testing.assert_array_equal(leaf.model.coef_, ref.coef_)
