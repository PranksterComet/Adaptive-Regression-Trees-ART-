"""CPU integration tests and optional numerical tests for the GPU QR optimizer."""

import builtins
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from art._cuda_optimizer import make_qr_workspace
from art.models import AffineRidgeModel, PolynomialRidgeModel, solve_weighted_ridge
from art.objectives import SoftObliqueRidgeObjective
from art.splitters import SoftObliqueSplitter, SplitNotFoundError, project_unit_w, project_unit_w_gradient
from art.timing import SplitterTimingProfile


@pytest.mark.parametrize("max_backtracks", [0, 1, 25])
def test_default_backtrack_batch_size_is_eight(max_backtracks):
    splitter = SoftObliqueSplitter(AffineRidgeModel(), 1., max_backtracks=max_backtracks)
    assert splitter.backtrack_batch_size == 8
    X, y = np.ones((8, 2)), np.ones(8)
    workspace = make_qr_workspace(splitter, X, y)
    assert workspace.batch_size == 8
    assert splitter.Xw_refresh == 32


@pytest.mark.parametrize("options,message", [
    ({"refit_during_line_search": True}, "refit_during_line_search=False"),
    ({"backtrack_batch_size": 0}, "backtrack_batch_size"),
    ({"backtrack_batch_size": 1.5}, "backtrack_batch_size"),
    ({"Xw_refresh": -1}, "Xw_refresh"),
    ({"weight_floor": np.nan}, "weight_floor"),
    ({"rho": 1}, "rho"),
    ({"max_backtracks": -1}, "max_backtracks"),
])
def test_invalid_gpu_options_fail_before_cupy_import(monkeypatch, options, message):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "cupy":
            raise AssertionError("Invalid options should fail before loading CUDA")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    splitter = SoftObliqueSplitter(AffineRidgeModel(), 1., optimizer="cupy_qr", **options)
    with pytest.raises(ValueError, match=message):
        splitter.split(np.ones((8, 2)), np.ones(8))


def test_lazy_workspace_and_stale_data_validation():
    X, y = np.arange(24.).reshape(12, 2), np.arange(12.)
    splitter = SoftObliqueSplitter(PolynomialRidgeModel(), 1., optimizer="cupy_qr")
    prepared = splitter.model_template.prepare_design(X)
    ws = make_qr_workspace(splitter, X, y, prepared)
    ws.validate(splitter, X.copy(), y.copy(), prepared)
    np.testing.assert_array_equal(ws.design, prepared.matrix)
    assert ws.cp is None
    for other_X, other_y in ((X + 1, y), (X, y + 1)):
        with pytest.raises(ValueError, match="different data"):
            ws.validate(splitter, other_X, other_y, prepared)
    with pytest.raises(ValueError, match="different data"):
        ws.validate(replace(splitter, backtrack_batch_size=2), X, y, prepared)


def test_qr_dispatch_is_separate_from_the_hybrid_objective(monkeypatch):
    import art._cuda_optimizer as gpu
    import art.splitters as splitters

    X, y = np.arange(24.).reshape(12, 2), np.arange(12.)
    marker, workspace = object(), object()
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(args)
        return marker

    def unexpected(*args, **kwargs):
        raise AssertionError("Full GPU QR must not construct the hybrid/CPU objective")

    monkeypatch.setattr(gpu, "split_cuda_qr", dispatch)
    monkeypatch.setattr(splitters, "SoftObliqueRidgeObjective", unexpected)
    splitter = SoftObliqueSplitter(AffineRidgeModel(solver="qr"), 1., optimizer="cupy_qr")
    assert splitter.split(X, y, qr_workspace=workspace) is marker
    assert calls[0][-1] is workspace


@pytest.mark.parametrize("backend,factory,dispatcher", [
    ("cupy_qr", "make_qr_workspace", "split_cuda_qr"),
    ("cupy_cholesky", "make_cholesky_workspace", "split_cuda_cholesky"),
])
def test_builder_reuses_qr_workspace_across_retries_without_serializing_it(monkeypatch, tmp_path, backend, factory, dispatcher):
    import joblib
    import art._cuda_optimizer as gpu
    from art.builder import RegressionTreeBuilder
    from art.domain import BoxDomain

    workspaces, attempts = [], []

    def create(*args):
        # A local lambda would make serialization fail if retained in the tree.
        workspace = lambda: None
        workspaces.append(workspace)
        return workspace

    def dispatch(splitter, X, y, rng, parent_loss, required, parent_solve, prepared, workspace,
                 *, hard_refit=True):
        attempts.append(workspace)
        if len(attempts) == 1:
            raise SplitNotFoundError("min_side_points", "force a retry")
        # Test control/data ownership with a CPU solve; GPU numerics are tested below.
        return replace(splitter, optimizer="cpu", random_state=4).split(
            X, y, parent_loss=parent_loss, prepared_design=prepared,
            hard_refit=hard_refit,
        )

    monkeypatch.setattr(gpu, factory, create)
    monkeypatch.setattr(gpu, dispatcher, dispatch)
    model = AffineRidgeModel(solver="qr")
    builder = RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1., 1.], [-1., 1.]])),
        oracle=lambda x: (1 if x[0] > 0 else -1) + .2 * x[1],
        model_template=model,
        splitter=SoftObliqueSplitter(model.clone(), .3, max_iters=3, optimizer=backend),
        sample_count=60, max_depth=1, error_threshold=0., max_retries_on_failure=1, random_state=5,
    )
    tree = builder.build().tree
    assert len(workspaces) == 1
    assert attempts == [workspaces[0], workspaces[0]]
    path = tmp_path / "qr_tree.joblib"
    joblib.dump(tree, path)
    restored = joblib.load(path)
    probe = np.array([[.2, .1], [-.7, .4]])
    np.testing.assert_array_equal(restored.predict(probe), tree.predict(probe))


@pytest.mark.parametrize("options", [
    {"max_iters": 0}, {"grad_atol": 1e10}, {"adaptive_alpha": False}, {},
    {"alpha0": 1e20, "max_backtracks": 0, "adaptive_alpha": False},
    {"alpha0": 1e20, "alpha_max": 1e20, "max_backtracks": 0},
])
def test_gpu_control_loop_matches_existing_stopping_and_step_rules_on_cpu(options):
    """Exercise the new controller with the existing CPU objective as its workspace.

    CUDA arithmetic is tested separately. This checks the actual new Python loop,
    including zero iterations, gradient stops, failed searches, and adaptive steps.
    """
    from art._cuda_optimizer import _optimize
    from art.optimizers import armijo_backtracking

    rng = np.random.default_rng(310)
    X = rng.uniform(-1, 1, (120, 3))
    y = np.where(X[:, 0] > 0, X[:, 1] + 2., .5 * X[:, 2] - 1.)
    splitter = SoftObliqueSplitter(AffineRidgeModel(1e-5, solver="qr"), .4, max_iters=5)
    splitter = replace(splitter, **options)
    objective = SoftObliqueRidgeObjective(.4, splitter.model_template, X, y)
    theta0 = splitter._initialize_theta(X, np.random.default_rng(5))

    class CpuWorkspace:
        n = len(X)
        x, cp = X, np
        timers = SimpleNamespace(measure=lambda *args, **kwargs: nullcontext())

        def start(self, theta, temperature):
            self.theta = project_unit_w(theta)
            self.last_feasible = np.empty_like(theta)
            self.v = np.empty(len(X))

        def current(self, temperature):
            self.evaluation = objective.value_and_grad(self.theta)
            self.pw = -project_unit_w_gradient(self.theta, self.evaluation.grad)
            left = self.evaluation.left_model.solve_result_.ridge_effective
            right = self.evaluation.right_model.solve_result_.ridge_effective
            return np.array([self.evaluation.loss, np.linalg.norm(self.pw),
                             self.evaluation.grad @ self.pw,
                             np.count_nonzero(self.evaluation.pi >= .5), left, right, 0])

        def search(self, splitter, alpha):
            result = armijo_backtracking(
                lambda t: objective.value(objective.evaluate(t, reference=self.evaluation)),
                lambda a: project_unit_w(self.theta + a * self.pw),
                self.evaluation.loss, self.evaluation.grad @ self.pw,
                alpha, splitter.rho, splitter.armijo_c, splitter.max_backtracks,
            )
            logical = result.n_backtracks + 1
            physical = min(splitter.max_backtracks + 1,
                           ((logical + splitter.backtrack_batch_size - 1) // splitter.backtrack_batch_size)
                           * splitter.backtrack_batch_size)
            return result, physical, logical

        def advance(self, alpha, temperature, refresh):
            self.theta = project_unit_w(self.theta + alpha * self.pw)

        def gate_summary(self):
            return {key: self.evaluation.metadata[key] for key in ("mean_pi", "min_pi", "max_pi")}

        def hard_result(self, splitter, parent_loss, required, converged, metadata, prepared):
            return splitter._make_hard_result(objective, self.theta, parent_loss,
                                              converged, required, metadata)

    expected = splitter._optimize_from_initialization(objective, theta0, 10., 2, None)
    actual = _optimize(splitter, CpuWorkspace(), theta0, 10., 2, None, None)
    np.testing.assert_array_equal(actual.w, expected.w)
    assert actual.z == expected.z
    assert actual.stop_reason == expected.stop_reason
    assert actual.converged == expected.converged
    for key in ("soft_loss_history", "line_search_loss_history", "step_size_history",
                "backtrack_history", "final_alpha", "initial_grad_norm", "final_grad_norm"):
        assert actual.metadata[key] == expected.metadata[key]


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if not count:
        pytest.skip("CUDA device unavailable")
    from art._cuda_qr import cuda_qr_status

    available, message = cuda_qr_status()
    assert available, message  # A present GPU with broken QR must fail, not silently skip.
    return cp


@pytest.mark.parametrize("optimizer", ["cupy_qr", "cupy_cholesky"])
def test_single_gpu_optimizer_can_skip_final_partition_fit(gpu, monkeypatch, optimizer):
    from art._cuda_optimizer import make_cholesky_workspace

    X = np.random.default_rng(913).uniform(-.5, .5, (20, 2))
    X[-1] = [2., 3.]
    y = np.sin(X[:, 0]) + X[:, 1]**2
    model = AffineRidgeModel(ridge=0., solver="auto")
    parent = model.clone().fit(X, y)
    s = SoftObliqueSplitter(model, 1., optimizer=optimizer, max_iters=0)
    monkeypatch.setattr(s, "_initialize_theta", lambda X, rng: np.array([1., 0., 1.]))
    factory = make_qr_workspace if optimizer == "cupy_qr" else make_cholesky_workspace
    ws = factory(s, X, y)
    fit = ws.fit

    def soft_fit():
        assert not ws.hard_weights, "Final hard fit must be skipped"
        return fit()

    monkeypatch.setattr(ws, "fit", soft_fit)
    options = {"qr_workspace" if optimizer == "cupy_qr" else "cholesky_workspace": ws}
    result = s.split(X, y, parent_model=parent, parent_loss=np.mean((parent.predict(X) - y)**2),
                      hard_refit=False, **options)
    assert result.n_left == 19 and result.n_right == 1
    assert not result.has_hard_fit and result.right_model is None


@pytest.mark.parametrize("ridge", [0., 1e-4])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("parallel", [False, True])
def test_gpu_current_matches_cpu_qr_fit_loss_and_projected_gradient(gpu, ridge, degree, parallel):
    rng = np.random.default_rng(311)
    X, y = rng.normal(size=(101, 3)), rng.normal(size=101)
    model = AffineRidgeModel(ridge, solver="qr") if degree == 1 else PolynomialRidgeModel(ridge=ridge, solver="qr")
    prepared = model.prepare_design(X) if degree == 2 else None
    splitter = SoftObliqueSplitter(model, .6, optimizer="cupy_qr", qr_parallel=parallel, weight_floor=.07)
    ws = make_qr_workspace(splitter, X, y, prepared)
    ws.initialize(None, None)
    theta = project_unit_w(rng.normal(size=4))
    objective = SoftObliqueRidgeObjective(.6, model, X, y, weight_floor=.07, prepared_design=prepared)
    expected = objective.value_and_grad(theta)
    with ws.stream:
        ws.start(theta, .6)
        packet = ws.current(.6)
        p = ws.pw.get(stream=ws.stream)
        np.testing.assert_allclose(packet[0], expected.loss, rtol=2e-12)
        np.testing.assert_allclose(p, -project_unit_w_gradient(theta, expected.grad), rtol=2e-10, atol=2e-13)
        for side, model in enumerate((expected.left_model, expected.right_model)):
            np.testing.assert_allclose(ws.plans[side].coefficients.get(stream=ws.stream), model.coef_, rtol=2e-10, atol=2e-13)


@pytest.mark.parametrize("batch_size", [8, 1, 4, 50])
@pytest.mark.parametrize("refresh", [0, 1, 3])
def test_gpu_optimizer_matches_cpu_qr_and_serializes(gpu, batch_size, refresh, tmp_path):
    import joblib

    rng = np.random.default_rng(312)
    X = rng.uniform(-1, 1, (160, 3))
    y = np.where(X[:, 0] > .1, X[:, 1] + 2., .5 * X[:, 2] - 1.)
    model = PolynomialRidgeModel(ridge=1e-5, solver="qr")
    cpu = SoftObliqueSplitter(model, .4, max_iters=6, random_state=4, alpha0=.2)
    expected = cpu.split(X, y)
    timing = SplitterTimingProfile()
    actual = replace(cpu, optimizer="cupy_qr", backtrack_batch_size=batch_size,
                     Xw_refresh=refresh, split_timing_profile=timing).split(X, y)
    np.testing.assert_allclose(actual.w, expected.w, atol=1e-9)
    np.testing.assert_allclose(actual.z, expected.z, atol=1e-9)
    np.testing.assert_allclose(actual.predict(X), expected.predict(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.metadata["soft_loss_history"], expected.metadata["soft_loss_history"], rtol=2e-9)
    assert actual.metadata["step_size_history"] == expected.metadata["step_size_history"]
    assert actual.metadata["backtrack_history"] == expected.metadata["backtrack_history"]
    assert actual.metadata["physical_line_search_candidates"] == timing.line_search_candidate_count
    assert actual.metadata["physical_line_search_candidates"] >= actual.metadata["logical_line_search_candidates"]
    assert actual.metadata["backtrack_batch_size"] == batch_size
    assert actual.metadata["Xw_refresh"] == refresh
    if batch_size >= cpu.max_backtracks + 1:
        assert timing.line_search_candidate_count % (cpu.max_backtracks + 1) == 0
    assert timing.least_squares_count == actual.n_iters + 2
    path = tmp_path / "split.joblib"
    joblib.dump(actual, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X), actual.predict(X))
    assert actual.left_model.solve_result_.solver_used == "qr"
    assert actual.left_model.solve_result_.cond_estimate is None


def test_gpu_rank_deficient_unregularized_design_reports_failure(gpu):
    rng = np.random.default_rng(313)
    X = rng.normal(size=(50, 2))
    X[:, 1] = X[:, 0]
    splitter = SoftObliqueSplitter(AffineRidgeModel(0., solver="qr"), 1., optimizer="cupy_qr")
    with pytest.raises(SplitNotFoundError, match="rank loss") as caught:
        splitter.split(X, rng.normal(size=50))
    assert caught.value.reason == "qr_failure"


def test_gpu_batched_search_matches_scalar_armijo_and_cached_advance(gpu):
    from art.optimizers import armijo_backtracking

    rng = np.random.default_rng(314)
    X, y = rng.normal(size=(111, 3)), rng.normal(size=111)
    splitter = SoftObliqueSplitter(AffineRidgeModel(0., solver="qr"), .7, optimizer="cupy_qr")
    theta = project_unit_w(rng.normal(size=4))
    objective = SoftObliqueRidgeObjective(.7, splitter.model_template, X, y)
    current = objective.value_and_grad(theta)
    p = -project_unit_w_gradient(theta, current.grad)
    expected = armijo_backtracking(lambda t: objective.value(objective.evaluate(t, reference=current)),
                                  lambda a: project_unit_w(theta + a * p), current.loss,
                                  current.grad @ p, alpha0=1e5)
    ws = make_qr_workspace(splitter, X, y)
    ws.initialize(None, None)
    with ws.stream:
        ws.start(theta, .7)
        ws.current(.7)
        gpu.matmul(ws.x, ws.pw[:-1], out=ws.v)
        actual, physical, logical = ws.search(splitter, 1e5)
        assert actual.success == expected.success
        assert actual.step_size == expected.step_size
        assert actual.n_backtracks == expected.n_backtracks
        assert actual.value == pytest.approx(expected.value, rel=1e-12)
        assert physical >= logical == expected.n_backtracks + 1
        if actual.success:
            ws.advance(actual.step_size, .7, refresh=False)
            t = ws.theta.get(stream=ws.stream)
            np.testing.assert_allclose(ws.u.get(stream=ws.stream), X @ t[:-1], atol=2e-14)


@pytest.mark.parametrize("problem", ["zero_norm", "nonfinite_norm", "nonfinite_score"])
def test_gpu_batched_search_rejects_invalid_trials(gpu, problem):
    """Exercise infinity sentinels in the real NVRTC-compiled trial kernel."""
    rng = np.random.default_rng(316)
    X, y = rng.normal(size=(33, 2)), rng.normal(size=33)
    splitter = SoftObliqueSplitter(
        AffineRidgeModel(solver="qr"), 1., optimizer="cupy_qr",
    )
    ws = make_qr_workspace(splitter, X, y)
    ws.initialize(None, None)
    with ws.stream:
        ws.start(np.array([1., 0., 0.]), 1.)
        ws.current(1.)
        gpu.matmul(ws.x, ws.pw[:-1], out=ws.v)
        if problem == "zero_norm":
            ws.geometry[:3] = 0.
        elif problem == "nonfinite_norm":
            ws.geometry[0] = gpu.inf
        else:
            ws.v.fill(gpu.inf)
        result, physical, logical = ws.search(splitter, 1.)
    assert not result.success
    assert np.isposinf(result.value)
    assert result.n_backtracks == splitter.max_backtracks
    assert physical == logical == splitter.max_backtracks + 1


def test_gpu_qr_plan_matches_augmented_numpy_solve(gpu):
    from art._cuda_qr import QrPlan

    rng = np.random.default_rng(315)
    phi, y = rng.normal(size=(40, 6)), rng.normal(size=40)
    weights, ridge = rng.uniform(size=40), 1e-3
    expected = solve_weighted_ridge(phi, y, weights, ridge, solver="qr")
    a = np.asfortranarray(np.vstack([phi * np.sqrt(weights[:, None]),
                                     np.sqrt(expected.ridge_effective) * np.eye(6)]))
    b = np.r_[y * np.sqrt(weights), np.zeros(6)]
    stream = gpu.cuda.Stream(non_blocking=True)
    plan = QrPlan(gpu, 46, 6, stream)
    try:
        with stream:
            plan.a.set(a)
            plan.b.set(b)
            plan.solve()
            np.testing.assert_allclose(plan.coefficients.get(stream=stream), expected.coefficients,
                                       rtol=1e-11, atol=1e-13)
            np.testing.assert_array_equal(plan.info.get(stream=stream), [0, 0])
    finally:
        stream.synchronize()
        plan.close()
