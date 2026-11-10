"""End-to-end multi-node optimization and pointer-prefix correctness checks."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from cuda_cpu_backend import cuda_backend  # noqa: F401
from art._cuda_batched_cholesky import BatchedCholeskyWorkspace
from art._cuda_batched_matmul import BatchedMatmul
from art._cuda_batched_optimizer import BatchedOptimizerWorkspace, _partition
from art.models import AffineRidgeModel, PolynomialRidgeModel, solve_weighted_ridge
from art.objectives import SoftObliqueRidgeObjective
from art.optimizers import AdaptiveAlpha, armijo_backtracking
from art.splitters import SoftObliqueSplitter, SplitNotFoundError, project_unit_w, project_unit_w_gradient


def problems():
    rng = np.random.default_rng(801)
    X = rng.uniform(-1, 1, (3, 43, 2))
    y = np.where(X[:, :, 0] > .1, 1. + X[:, :, 1], -.5 + .3 * X[:, :, 1])
    y[0] = 0.  # Exact fit: stops before any GD step while other nodes continue.
    theta = np.tile([1., 0., 0.], (3, 1))
    return X, y, theta


def reference(s, X, y, theta, temperature, trace=None):
    """Independent CPU objective + existing scalar line-search/controller."""
    objective = SoftObliqueRidgeObjective(temperature, s.model_template, X, y,
                                         weight_floor=s.weight_floor)
    theta = project_unit_w(theta)
    if trace is not None:
        trace.append(theta.copy())
    current = objective.value_and_grad(theta)
    losses, steps, backs = [current.loss], [], []
    norm0 = np.linalg.norm(project_unit_w_gradient(theta, current.grad))
    tolerance = s.grad_atol + s.grad_rtol * norm0
    controller = AdaptiveAlpha(s.alpha0, s.alpha_min, s.alpha_max, s.alpha_grow,
                               s.alpha_recovery, s.heavy_backtrack_threshold)
    reason = "max_iters"
    for _ in range(s.max_iters):
        p = -project_unit_w_gradient(theta, current.grad)
        if np.linalg.norm(p) <= tolerance:
            reason = "gradient_tolerance"
            break
        if current.grad @ p >= 0:
            reason = "non_descent_direction"
            break
        failures = 0
        while True:
            search = armijo_backtracking(
                lambda t: objective.value(objective.evaluate(t, reference=current)),
                lambda a: project_unit_w(theta + a * p), current.loss, current.grad @ p,
                controller.alpha if s.adaptive_alpha else s.alpha0, s.rho, s.armijo_c, s.max_backtracks)
            if s.adaptive_alpha:
                controller.update(search, s.rho)
            if search.success:
                break
            failures += 1
            stop = ("line_search_failed" if not s.adaptive_alpha else
                    "max_line_search_failures" if failures >= s.max_line_search_failures else
                    "alpha_min_reached" if controller.alpha <= s.alpha_min else None)
            if stop:
                reason = stop
                break
        if not search.success:
            break
        theta = search.theta
        if trace is not None:
            trace.append(theta.copy())
        current = objective.value_and_grad(theta)
        losses.append(current.loss); steps.append(search.step_size); backs.append(search.n_backtracks)
    return theta, losses, steps, backs, reason


def test_active_prefix_partition_handles_converged_tail_entries():
    ids = np.arange(6, dtype=np.int32)
    keep = np.array([True, False, True, False, True, False])
    count = _partition(ids, keep)
    assert count == 3
    assert keep[ids[:count]].all() and not keep[ids[count:]].any()
    np.testing.assert_array_equal(np.sort(ids), np.arange(6))


@pytest.mark.parametrize("degree,reorder", [(1, False), (1, True), (2, False), (2, True)])
def test_partition_export_skips_singular_hard_refit(cuda_backend, monkeypatch, degree, reorder):
    rng = np.random.default_rng(912)
    X = rng.uniform(-.5, .5, (2, 20, 2))
    X[:, -1] = [2., 3.]  # A one-point right partition has a singular hard Gram.
    y = np.sin(X[:, :, 0]) + X[:, :, 1]**2
    model = AffineRidgeModel(ridge=0.) if degree == 1 else PolynomialRidgeModel(ridge=0.)
    s = SoftObliqueSplitter(model, 1., max_iters=0)
    theta = np.tile([1., 0., 1.], (2, 1))
    parent = np.array([np.mean((model.clone().fit(x, z).predict(x) - z)**2) for x, z in zip(X, y)])
    with BatchedOptimizerWorkspace(s, X, y, reorder=reorder) as ws:
        fitted = ws.optimize(theta, parent_loss=parent)
        assert all(isinstance(r, SplitNotFoundError) and r.reason == "cholesky_failure" for r in fitted)
        fit = ws._fit

        def soft_fit(**kwargs):
            assert not kwargs.get("hard", False), "Hard refit was not skipped"
            return fit(**kwargs)

        monkeypatch.setattr(ws, "_fit", soft_fit)
        results = ws.optimize(theta, parent_loss=parent, hard_refit=False)
        assert ws.fit_sizes == [("soft", 2)]
        for result in results:
            assert not isinstance(result, SplitNotFoundError), result
            assert result.n_left == 19 and result.n_right == 1
            assert result.left_model is result.right_model is None
            assert result.loss is result.split_gain is result.relative_split_gain is None
            assert result.metadata["hard_refit"] is False
            assert result.metadata["n_right"] == 1
            np.testing.assert_array_equal(np.r_[result.w, result.z], theta[0])


@pytest.mark.parametrize("layout", ["row", "transpose", "padded", "vector"])
def test_pointer_matmul_reads_selected_slots_without_copying_matrices(cuda_backend, layout):
    cp, rng = cuda_backend, np.random.default_rng(802)
    from cupy_backends.cuda.libs import cublas

    a = cp.asarray(rng.normal(size=(4, 7, 3)))
    if layout == "transpose":
        a = a.swapaxes(-1, -2)
    if layout == "padded":
        a = cp.asarray(rng.normal(size=(4, 7, 5)))[:, :, :3]
    b = cp.asarray(rng.normal(size=(4, a.shape[2], 1 if layout == "vector" else 2)))
    out = cp.array(np.full((4, a.shape[1], b.shape[2]), -7.))
    handle = cublas.create()
    stream = cp.cuda.get_current_stream()
    try:
        plan = BatchedMatmul(cp, a, b, out, handle, stream)
        for ids in ([3, 1], [2], []):
            out.fill(-7.)
            plan.select(cp.asarray(np.array(ids, dtype=np.int32)), len(ids))
            plan.run()
            stream.synchronize()
            expected = np.full(out.shape, -7.)
            for j in ids:
                expected[j] = a.get()[j] @ b.get()[j]
            np.testing.assert_allclose(out.get(), expected, rtol=1e-12, atol=1e-13)
    finally:
        stream.synchronize(); cublas.destroy(handle)


@pytest.mark.parametrize("p", [3, 17])
def test_subset_fits_preserve_inactive_coefficients_and_allow_hard_floor_override(cuda_backend, p):
    rng = np.random.default_rng(803)
    n = 2 * p + 3
    phi, y = rng.normal(size=(4, n, p)), rng.normal(size=(4, n))
    pi = rng.uniform(size=(4, n))
    with BatchedCholeskyWorkspace(phi, y, ridge=.02, weight_floor=.1) as ws:
        ws.fit(pi); ws.synchronize()
        old_beta, old_status = ws.coefficients.get(), ws.status.get()
        phi_ptr, y_ptr = ws.phi.data.ptr, ws.y.data.ptr
        pi[0, 0] = np.nan  # Unselected bad gate must never affect another node.
        pi[[3, 1]] = (pi[[3, 1]] >= .5).astype(float)
        pi[[3, 1], 0], pi[[3, 1], 1] = 0., 1.
        if hasattr(cuda_backend, "calls"):
            cuda_backend.calls.clear()
        ws.fit(pi, nodes=[3, 1], weight_floor=0.)
        ws.synchronize()
        assert ws.last_batch_size == 4
        if hasattr(cuda_backend, "calls"):
            assert [call[1] for call in cuda_backend.calls if call[0] == "gemm"] == [2] * 4
            assert [call for call in cuda_backend.calls if call[0] == "potrf"] == [("potrf", 4)]
        for j in (3, 1):
            for side, q in enumerate((1 - pi[j], pi[j])):
                ref = solve_weighted_ridge(phi[j], y[j], q, .02, weight_floor=0., solver="normal")
                np.testing.assert_allclose(ws.coefficients.get()[j, side], ref.coefficients, rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(ws.coefficients.get()[[0, 2]], old_beta[[0, 2]])
        np.testing.assert_array_equal(ws.status.get()[[0, 2]], old_status[[0, 2]])
        assert (ws.phi.data.ptr, ws.y.data.ptr) == (phi_ptr, y_ptr)
        ws.fit(pi, nodes=[])
        assert ws.last_batch_size == 0
        for bad in ([1, 1], [4], [-1], [1.5]):
            with pytest.raises(ValueError, match="nodes"):
                ws.fit(pi, nodes=bad)


def test_subset_solver_statuses_scatter_back_to_original_slots(cuda_backend):
    rng = np.random.default_rng(806)
    phi, y = rng.normal(size=(4, 20, 3)), rng.normal(size=(4, 20))
    phi[1, :, 2] = 0.
    pi = np.full((4, 20), .4)
    with BatchedCholeskyWorkspace(phi, y, ridge=0.) as ws:
        ws.fit(pi); ws.synchronize()
        before = ws.coefficients.get().copy()
        ws.fit(pi, nodes=[3, 1]); ws.synchronize()
        np.testing.assert_array_equal(ws.status.get()[[3, 1]], [[0, 0], [3, 3]])
        assert np.isnan(ws.coefficients.get()[1]).all()
        np.testing.assert_array_equal(ws.coefficients.get()[[0, 2]], before[[0, 2]])
        pi[1, 0] = np.nan
        ws.fit(pi, nodes=[1]); ws.synchronize()
        np.testing.assert_array_equal(ws.status.get()[1], [-1, -1])


@pytest.mark.parametrize("degree,adaptive,refresh,H", [(1, False, 0, 8), (1, True, 1, 1), (2, True, 2, 3)])
def test_both_modes_match_independent_node_optimizations(cuda_backend, degree, adaptive, refresh, H):
    X, y, theta = problems()
    model = AffineRidgeModel(1e-4, solver="normal") if degree == 1 else PolynomialRidgeModel(ridge=1e-4, solver="normal")
    s = SoftObliqueSplitter(model, .7, max_iters=4, alpha0=2., adaptive_alpha=adaptive,
                            min_side_points=2, backtrack_batch_size=H, Xw_refresh=refresh)
    temp = np.array([.3, .7, 1.2])
    outcomes, counts = [], []
    for reorder in (False, True):
        with BatchedOptimizerWorkspace(s, X, y, temperature=temp, reorder=reorder) as ws:
            results = ws.optimize(theta)
            counts.append(ws.ls.fit_count)
            assert all(not isinstance(r, SplitNotFoundError) for r in results), results
            assert any(count < len(X) for phase, count in ws.fit_sizes) == reorder
            for j, result in enumerate(results):
                expected, losses, steps, backs, reason = reference(s, X[j], y[j], theta[j], temp[j])
                np.testing.assert_allclose(np.r_[result.w, result.z], expected, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(result.metadata["soft_loss_history"], losses, rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(result.metadata["step_size_history"], steps, rtol=1e-13)
                assert result.metadata["backtrack_history"] == backs
                assert result.stop_reason == reason
                assert result.n_iters == len(steps)
                left = X[j] @ result.w < result.z
                phi = ws.ls.phi.get()[j]
                for side, q in enumerate((left.astype(float), (~left).astype(float))):
                    ref = solve_weighted_ridge(phi, y[j], q, model.ridge, weight_floor=0., solver="normal")
                    fitted = result.left_model if side == 0 else result.right_model
                    np.testing.assert_allclose(fitted.coef_, ref.coefficients, rtol=1e-10, atol=1e-12)
                assert result.metadata["physical_line_search_candidates"] >= result.metadata["logical_line_search_candidates"]
            assert results[0].n_iters == 0
            exported = pickle.loads(pickle.dumps(results[-1]))
            assert type(exported.w) is np.ndarray and type(exported.left_model.coef_) is np.ndarray
            np.testing.assert_allclose(exported.predict(X[-1]), results[-1].predict(X[-1]))
            outcomes.append(results)
    assert counts[1] < counts[0]
    for a, b in zip(*outcomes):
        np.testing.assert_allclose(a.predict(X[0]), b.predict(X[0]), rtol=1e-10, atol=1e-12)


def test_restarts_and_numerical_failures_are_isolated(cuda_backend):
    X, y, theta = problems()
    X[1, :, 1] = 0.  # Deterministic singular design without ridge.
    s = SoftObliqueSplitter(AffineRidgeModel(0., solver="normal"), .7,
                            max_iters=2, n_restarts=2, min_side_points=2)
    initial = np.stack((theta, np.tile([0., 1., 0.], (3, 1))))
    with BatchedOptimizerWorkspace(s, X, y, reorder=True) as ws:
        if hasattr(cuda_backend, "host_reads"):
            cuda_backend.host_reads.clear()
        result = ws.optimize(initial)
        if hasattr(cuda_backend, "host_reads"):
            assert cuda_backend.host_reads.count((3, 3)) == 1  # Export theta once, not each restart/step.
        assert isinstance(result[1], SplitNotFoundError)
        assert result[1].reason == "cholesky_failure"
        assert result[1].diagnostics["stop_reason"] == "cholesky_failure"
        assert result[1].diagnostics["optimizer"] == "cupy_batched_cholesky"
        assert all(not isinstance(result[j], SplitNotFoundError) for j in (0, 2))
        again = ws.optimize(initial)
        np.testing.assert_allclose(result[2].predict(X[2]), again[2].predict(X[2]), atol=1e-12)
    with pytest.raises(RuntimeError, match="closed"):
        ws.optimize(initial)


@pytest.mark.parametrize("adaptive", [False, True])
def test_line_search_exhaustion_preserves_theta_and_counts_retries(cuda_backend, adaptive):
    X, y, theta = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4, solver="normal"), .7,
                            max_iters=2, alpha0=1e12, alpha_max=1e12,
                            max_backtracks=0, max_line_search_failures=2, adaptive_alpha=adaptive)
    with BatchedOptimizerWorkspace(s, X, y) as ws:
        result = ws.optimize(theta)
        for j in (1, 2):
            _, _, steps, backs, reason = reference(s, X[j], y[j], theta[j], .7)
            assert result[j].stop_reason == reason
            np.testing.assert_allclose(result[j].metadata["step_size_history"], steps)
            assert result[j].metadata["backtrack_history"] == backs
            assert result[j].n_iters == 0
            np.testing.assert_allclose(np.r_[result[j].w, result[j].z], theta[j], atol=1e-14)
            assert result[j].metadata["physical_line_search_candidates"] == (2 if adaptive else 1)


@pytest.mark.parametrize("hard_refit", [False, True])
def test_last_feasible_parameters_are_restored_before_hard_refits(cuda_backend, hard_refit):
    data = []
    for seed in (2, 3):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (60, 2))
        data.append((x, (x[:, 0] > .55).astype(float)))
    X, y = np.stack([v[0] for v in data]), np.stack([v[1] for v in data])
    theta = np.tile([1., 0., 0.], (2, 1))
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4, solver="normal"), .25,
                            max_iters=6, alpha0=5., min_side_points=28, min_side_fallback=False)
    with BatchedOptimizerWorkspace(s, X, y) as ws:
        results = ws.optimize(theta, min_side_fallback=True, hard_refit=hard_refit)
        assert not s.min_side_fallback  # Final-attempt override is local to this call.
        for j, result in enumerate(results):
            assert not isinstance(result, SplitNotFoundError), result
            trace = []
            final, *_ = reference(s, X[j], y[j], theta[j], .25, trace)
            right = np.count_nonzero(X[j] @ final[:-1] >= final[-1])
            assert min(right, 60 - right) < 28
            feasible = [(i, t) for i, t in enumerate(trace)
                        if 28 <= np.count_nonzero(X[j] @ t[:-1] >= t[-1]) <= 32]
            iteration, expected = feasible[-1]
            np.testing.assert_allclose(np.r_[result.w, result.z], expected, rtol=1e-8, atol=1e-10)
            assert result.metadata["fallback_iteration"] == iteration
            assert result.metadata["min_side_fallback"]
            assert min(result.n_left, result.n_right) >= 28


def test_failed_attempts_can_retry_and_then_accept_independently(cuda_backend):
    X, y, theta = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4, solver="normal"), .7,
                            max_iters=2, alpha0=1000., armijo_c=.8,
                            max_backtracks=0, max_line_search_failures=20)
    with BatchedOptimizerWorkspace(s, X, y) as ws:
        results = ws.optimize(theta)
        for j in (1, 2):
            result = results[j]
            assert not isinstance(result, SplitNotFoundError), result
            expected, _, steps, backs, reason = reference(s, X[j], y[j], theta[j], .7)
            np.testing.assert_allclose(np.r_[result.w, result.z], expected, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(result.metadata["step_size_history"], steps)
            assert result.metadata["backtrack_history"] == backs
            assert result.stop_reason == reason
            assert result.n_iters == 2
            assert result.metadata["physical_line_search_candidates"] > 2


def test_single_slot_and_cached_parent_loss(cuda_backend):
    X, y, theta = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4, solver="normal"), 1., max_iters=0)
    with BatchedOptimizerWorkspace(s, X[2:], y[2:]) as ws:
        result = ws.optimize(theta[2:], parent_loss=np.array([.125]))[0]
        assert not isinstance(result, SplitNotFoundError)
        assert ws.fit_sizes == [("soft", 1), ("hard", 1)]
        assert result.parent_loss == .125
        assert result.split_gain == pytest.approx(.125 - result.loss)
        assert result.n_iters == 0 and result.stop_reason == "max_iters"
        np.testing.assert_allclose(np.r_[result.w, result.z], theta[2])


@pytest.mark.parametrize("degree,bias", [(1, True), (2, True), (2, False)])
def test_gpu_features_are_built_once_and_shared_without_cpu_transform(cuda_backend, monkeypatch, degree, bias):
    from sklearn.preprocessing import PolynomialFeatures
    from art import _cuda_batched_optimizer as driver

    X, y, theta = problems()
    model = PolynomialRidgeModel(degree=degree, include_bias=bias, ridge=1e-4)
    s = SoftObliqueSplitter(model, 1., max_iters=0, min_side_points=2)
    expected = PolynomialFeatures(degree=degree, include_bias=bias).fit_transform(
        X.reshape(-1, X.shape[2])).reshape(len(X), X.shape[1], -1)
    built, original = [], driver.FeaturePlan.build

    def build(*args, **kwargs):
        phi = original(*args, **kwargs)
        built.append(phi)
        return phi

    def forbidden(*args, **kwargs):
        raise AssertionError("Workspace must not transform features on CPU")

    monkeypatch.setattr(driver.FeaturePlan, "build", build)
    # Only transformer.fit() for tiny dimension metadata is allowed on CPU.
    with monkeypatch.context() as patch:
        patch.setattr(PolynomialRidgeModel, "prepare_design", forbidden)
        patch.setattr(PolynomialFeatures, "transform", forbidden)
        patch.setattr(PolynomialFeatures, "fit_transform", forbidden)
        ws = BatchedOptimizerWorkspace(s, X, y)
    with ws:
        assert len(built) == 1 and ws.ls.phi.data.ptr == built[0].data.ptr
        ws.stream.synchronize()
        np.testing.assert_array_equal(ws.ls.phi.get(), expected)
        phi_ptr = ws.ls.phi.data.ptr
        for _ in range(2):
            results = ws.optimize(theta)
            for j, result in enumerate(results):
                assert not isinstance(result, SplitNotFoundError), result
                np.testing.assert_allclose(result.left_model.predict(X[j]),
                                           expected[j] @ result.left_model.coef_, rtol=1e-12, atol=1e-13)
        assert len(built) == 1 and ws.ls.phi.data.ptr == phi_ptr


def test_capacity_cholesky_reloads_without_moving_storage_or_fitting_padding(cuda_backend):
    cp, rng = cuda_backend, np.random.default_rng(811)
    with BatchedCholeskyWorkspace.allocate(5, 23, 4, ridge=.02) as ws:
        with pytest.raises(RuntimeError, match="Load"):
            ws.fit(np.ones((5, 23)))
        def buffers():
            return [ws.phi, ws.y, ws.weights, ws.grams, ws._factor_storage, ws._rhs,
                    ws._a_ptrs, ws._ap, ws.ids, ws._products[0][0].base]

        addresses = [a.data.ptr for a in buffers()]
        for k in (3, 1, 5, 2):
            phi, y = rng.normal(size=(k, 23, 4)), rng.normal(size=(k, 23))
            pi = rng.uniform(size=y.shape)
            ws.load(phi, y)
            ws.phi[k:].fill(np.nan)
            ws.y[k:].fill(np.nan)
            if hasattr(cp, "calls"):
                cp.calls.clear()
            ws.fit(pi)
            ws.synchronize()
            assert ws.K == 5 and ws.k == k and ws.last_batch_size == 2 * k
            np.testing.assert_array_equal(ws.status.get()[:k], 0)
            if hasattr(cp, "calls"):
                assert [(c[0], c[1]) for c in cp.calls if c[0] == "potrf"] == [("potrf", 2 * k)]
            actual = ws.coefficients.get()
            for j in range(k):
                for side, q in enumerate((1. - pi[j], pi[j])):
                    ref = solve_weighted_ridge(phi[j], y[j], q, .02, solver="normal")
                    np.testing.assert_allclose(actual[j, side], ref.coefficients, rtol=3e-11, atol=1e-12)
            assert addresses == [a.data.ptr for a in buffers()]
        bad = phi.copy(); bad[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            ws.load(bad, y)
        assert ws.k == 0
        with pytest.raises(RuntimeError, match="Load"):
            ws.fit(pi)
    assert ws.phi is None and ws._products is None


@pytest.mark.parametrize("reorder", [False, True])
def test_optimizer_capacity_loads_match_independent_cpu_and_preserve_exports(cuda_backend, reorder):
    X, y, theta = problems()
    s = SoftObliqueSplitter(PolynomialRidgeModel(ridge=1e-4), .7, max_iters=2,
                            min_side_points=2, alpha0=.5, max_backtracks=4)
    with BatchedOptimizerWorkspace.allocate(s, 5, X.shape[1], X.shape[2], reorder=reorder) as ws:
        with pytest.raises(RuntimeError, match="Load"):
            ws.optimize()
        def buffers():
            return [ws.x, ws.ls.y, ws.ls.phi, ws.ls._rhs, ws.ls._factor_storage,
                    ws.features.a, ws.features.b, ws.products["pred"].base,
                    ws.ls._products[0][0].base, ws.packet, ws.trial_partials]

        addresses = [a.data.ptr for a in buffers()]
        host_ids = [id(a) for a in (ws.host_X, ws.host_y, ws._packet_host, ws._search_host)]
        previous = None
        for batch, k in enumerate((3, 1, 2)):
            take = [2, 1, 0][:k]
            batch_X, batch_y = X[take] + .07 * batch, y[take] + .2 * batch
            batch_theta = theta[take]
            # Pack straight into the workspace's persistent host tensors.
            ws.host_X[:k], ws.host_y[:k] = batch_X, batch_y
            temps = np.linspace(.5, 1., k)
            ws.load(ws.host_X[:k], ws.host_y[:k], temperature=temps)
            ws.x[k:].fill(np.nan); ws.ls.phi[k:].fill(np.nan); ws.ls.y[k:].fill(np.nan)
            parent = None
            if batch == 1:
                parent = []
                for x, target in zip(batch_X, batch_y):
                    model = s.model_template.clone().fit(x, target)
                    parent.append(np.mean((model.predict(x) - target)**2))
                parent = np.array(parent)
            results = ws.optimize(batch_theta, parent_loss=parent)
            assert len(results) == k and ws.K == 5 and ws.k == k
            assert all(count <= k for _, count in ws.fit_sizes)
            if not reorder:
                assert all(count == k for _, count in ws.fit_sizes)
            assert any(phase == "parent" for phase, _ in ws.fit_sizes) == (parent is None)
            for j, result in enumerate(results):
                assert not isinstance(result, SplitNotFoundError), result
                expected, losses, steps, backs, reason = reference(
                    s, batch_X[j], batch_y[j], batch_theta[j], temps[j])
                np.testing.assert_allclose(np.r_[result.w, result.z], expected, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(result.metadata["soft_loss_history"], losses, rtol=1e-9, atol=1e-12)
                assert result.stop_reason == reason and result.metadata["backtrack_history"] == backs
            if previous is not None:
                old, predictions = previous
                np.testing.assert_array_equal(old.predict(X[0]), predictions)
            previous = results[0], results[0].predict(X[0]).copy()
            assert addresses == [a.data.ptr for a in buffers()]
            assert host_ids == [id(a) for a in (ws.host_X, ws.host_y, ws._packet_host, ws._search_host)]
    assert ws.x is None and ws.host_X is None and not ws.products
    with pytest.raises(RuntimeError, match="closed"):
        ws.load(X, y)


@pytest.mark.parametrize("reorder", [False, True])
def test_selected_retry_reuses_features_and_preserves_slot_order(cuda_backend, monkeypatch, reorder):
    from art._cuda_features import FeaturePlan

    X, y, theta = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4), .7, max_iters=1,
                            min_side_points=2, min_side_fallback=True, random_state=28)
    calls, original = [], FeaturePlan.build

    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(FeaturePlan, "build", build)
    with BatchedOptimizerWorkspace.allocate(s, 5, X.shape[1], X.shape[2], reorder=reorder) as ws:
        ws.load(X, y)
        first = ws.optimize(theta)
        phi = ws.ls.phi.get().copy()
        retry = ws.optimize(theta[[2, 1]], nodes=[2, 1], min_side_fallback=False)
        assert calls == [1]
        assert ws.run_ids.tolist() == [2, 1] and not ws._fallback and s.min_side_fallback
        assert all(count == 2 for _, count in ws.fit_sizes)
        np.testing.assert_array_equal(ws.ls.phi.get(), phi)
        for result, j in zip(retry, (2, 1)):
            assert not isinstance(result, SplitNotFoundError), result
            np.testing.assert_allclose(np.r_[result.w, result.z], np.r_[first[j].w, first[j].z],
                                       rtol=1e-9, atol=1e-10)
        assert ws.optimize(nodes=[]) == []
        for ids in ([3], [1, 1], [-1], [1.2]):
            with pytest.raises(ValueError, match="nodes"):
                ws.optimize(nodes=ids)
        fresh = ws.optimize(nodes=[2], parent_loss=np.array([first[2].parent_loss]))[0]
        assert fresh.metadata["initialization"] == "gpu"
        assert ws._fallback and calls == [1]


def test_capacity_load_rejects_bad_data_and_recovers_after_feature_overflow(cuda_backend):
    X, y, _ = problems()
    s = SoftObliqueSplitter(PolynomialRidgeModel(), 1., max_iters=0)
    with BatchedOptimizerWorkspace.allocate(s, 3, X.shape[1], X.shape[2]) as ws:
        for bad_X, bad_y, temp in ((X[:0], y[:0], 1.), (X, y[:, :-1], 1.),
                                  (X, y, 0.), (X * np.nan, y, 1.)):
            with pytest.raises(ValueError):
                ws.load(bad_X, bad_y, temperature=temp)
        with np.errstate(over="ignore", invalid="ignore"):
            with pytest.raises(ValueError, match="finite"):
                ws.load(np.full_like(X, 1e200), y)
        assert ws.k == ws.ls.k == 0
        with pytest.raises(RuntimeError, match="Load"):
            ws.optimize()
        ws.load(X, y)
        assert ws.k == 3


def test_capacity_optimizer_runs_on_its_supplied_stream(cuda_backend):
    cp = cuda_backend
    if not hasattr(cp.cuda, "Stream"):
        pytest.skip("Non-default stream behavior requires actual CUDA")
    X, y, theta = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(1e-4), .7, max_iters=1, min_side_points=2)
    stream = cp.cuda.Stream(non_blocking=True)
    with BatchedOptimizerWorkspace.allocate(s, 5, 43, 2, stream=stream) as ws:
        ws.load(X, y)
        results = ws.optimize(theta)
        for j, result in enumerate(results):
            assert not isinstance(result, SplitNotFoundError), result
            expected, *_ = reference(s, X[j], y[j], theta[j], .7)
            np.testing.assert_allclose(np.r_[result.w, result.z], expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("change", ["shape", "nonfinite", "temperature", "reorder", "refit", "degree",
                                    "capacity", "n", "d"])
def test_invalid_configuration_is_rejected_before_gpu_import(change, monkeypatch):
    import builtins

    X, y, _ = problems()
    s = SoftObliqueSplitter(AffineRidgeModel(), .7)
    kwargs = {}
    if change == "shape": y = y[:, :-1]
    if change == "nonfinite": X[0, 0, 0] = np.nan
    if change == "temperature": kwargs["temperature"] = [1., 0., 1.]
    if change == "reorder": kwargs["reorder"] = "yes"
    if change == "refit": s = replace(s, refit_during_line_search=True)
    if change == "degree": s = replace(s, model_template=PolynomialRidgeModel(degree=3))
    original = builtins.__import__

    def guarded(name, *args, **kw):
        if name == "cupy" or name.startswith("cupy_backends"):
            raise AssertionError("Input validation must precede GPU imports")
        return original(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError):
        if change in ("capacity", "n", "d"):
            dims = dict(capacity=3, n=43, d=2)
            dims[change] = 0
            BatchedOptimizerWorkspace.allocate(s, **dims)
        else:
            BatchedOptimizerWorkspace(s, X, y, **kwargs)
