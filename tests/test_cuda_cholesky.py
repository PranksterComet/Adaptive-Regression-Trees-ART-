"""Integration and real-device checks for the resident Cholesky optimizer."""

import builtins
from dataclasses import replace

import numpy as np
import pytest

from art._cuda_optimizer import make_cholesky_workspace
from art.models import AffineRidgeModel, PolynomialRidgeModel, solve_weighted_ridge
from art.objectives import SoftObliqueRidgeObjective
from art.splitters import SoftObliqueSplitter, SplitNotFoundError, project_unit_w, project_unit_w_gradient
from art.timing import SplitterTimingProfile


def test_cholesky_workspace_is_lazy_and_validates_backend_and_data(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("Creating a lazy workspace must not import CuPy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    X, y = np.arange(36.).reshape(12, 3), np.arange(12.)
    splitter = SoftObliqueSplitter(AffineRidgeModel(), 1., optimizer="cupy_cholesky")
    ws = make_cholesky_workspace(splitter, X, y)
    assert ws.cp is None
    ws.validate(splitter, X.copy(), y.copy(), None)
    for other, xx in [(splitter, X + 1), (replace(splitter, optimizer="cupy_qr"), X)]:
        with pytest.raises(ValueError, match="different data or configuration"):
            ws.validate(other, xx, y, None)
    with pytest.raises(ValueError, match="hybrid_gram uses the CPU optimizer"):
        make_cholesky_workspace(replace(splitter, optimizer="hybrid_gram"), X, y)
    with pytest.raises(ValueError, match="refit_during_line_search=False"):
        make_cholesky_workspace(replace(splitter, refit_during_line_search=True), X, y)


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if not count:
        pytest.skip("CUDA device unavailable")
    from art._cuda_cholesky import cuda_cholesky_status

    ok, message = cuda_cholesky_status()
    assert ok, message
    return cp


def test_reusable_cholesky_plan_matches_cpu_normal_solve(gpu):
    from art._cuda_cholesky import CholeskyPlan

    rng = np.random.default_rng(401)
    phi, y, q = rng.normal(size=(40, 6)), rng.normal(size=40), rng.uniform(.1, 1., 40)
    expected = solve_weighted_ridge(phi, y, q, 1e-3, solver="normal")
    gram = phi.T @ (q[:, None] * phi) + expected.ridge_effective * np.eye(6)
    rhs = phi.T @ (q * y)
    stream = gpu.cuda.Stream(non_blocking=True)
    plan = CholeskyPlan(gpu, 6, stream)
    try:
        for _ in range(2):
            with stream:
                plan.a.set(np.asfortranarray(gram)); plan.b.set(rhs)
                plan.solve()
                np.testing.assert_allclose(plan.coefficients.get(stream=stream), expected.coefficients,
                                           rtol=1e-11, atol=1e-13)
                np.testing.assert_array_equal(plan.info.get(stream=stream), [0, 0])
    finally:
        stream.synchronize()
        plan.close()


@pytest.mark.parametrize("floor", [0., 1e-12, .07, .75])
@pytest.mark.parametrize("gate", ["mixed", "left", "right"])
def test_complementary_grams_preserve_independent_floors_and_direct_rhs(gpu, monkeypatch, floor, gate):
    rng = np.random.default_rng(402)
    X, y = rng.normal(size=(43, 3)), rng.normal(size=43)
    splitter = SoftObliqueSplitter(PolynomialRidgeModel(), 1., optimizer="cupy_cholesky", weight_floor=floor)
    prepared = splitter.model_template.prepare_design(X)
    ws = make_cholesky_workspace(splitter, X, y, prepared)
    ws.initialize(None, None)
    total = ws.gram_pair.total
    pi = rng.uniform(.1, .9, len(y))
    if gate == "mixed":
        pi[:4] = [0., 1., 1e-20, 1. - 1e-15]
    else:
        pi.fill(0. if gate == "left" else 1.)
    q = np.maximum([1. - pi, pi], floor)
    products = []
    original_matmul = gpu.matmul

    def observed(a, b, *args, **kwargs):
        if a.shape == (ws.p, ws.n) and b.shape == (ws.n, ws.p):
            products.append(True)
        return original_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(gpu, "matmul", observed)
    with ws.stream:
        ws.pi.set(pi); ws.q.set(q)
        gpu.matmul(ws.q, ws.energy, out=ws.traces)
        ws.gram_pair.assemble(ws.plans, ws.q, ws.pi, ws.traces, floor, ws.stream)
        for side, plan in enumerate(ws.plans):
            np.testing.assert_allclose(plan.a.get(stream=ws.stream), prepared.matrix.T @ (q[side, :, None] * prepared.matrix), rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(plan.b.get(stream=ws.stream), prepared.matrix.T @ (q[side] * y), rtol=2e-12, atol=2e-12)
    assert ws.gram_pair.total is total
    # With no floor correction the pair uses exactly one weighted Gram product.
    if floor == 0.:
        assert len(products) == 1


@pytest.mark.parametrize("ridge", [0., 1e-4])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("parallel", [False, True])
def test_current_matches_cpu_normal_coefficients_loss_gradient_and_ridge(gpu, ridge, degree, parallel):
    rng = np.random.default_rng(403)
    X, y = rng.normal(size=(101, 3)), rng.normal(size=101)
    model = AffineRidgeModel(ridge, solver="normal") if degree == 1 else PolynomialRidgeModel(ridge=ridge, solver="normal")
    prepared = model.prepare_design(X) if degree == 2 else None
    splitter = SoftObliqueSplitter(model, .6, optimizer="cupy_cholesky", qr_parallel=parallel, weight_floor=.07)
    ws = make_cholesky_workspace(splitter, X, y, prepared)
    ws.initialize(None, None)
    theta = project_unit_w(rng.normal(size=4))
    expected = SoftObliqueRidgeObjective(.6, model, X, y, weight_floor=.07, prepared_design=prepared).value_and_grad(theta)
    with ws.stream:
        ws.start(theta, .6)
        packet = ws.current(.6)
        np.testing.assert_allclose(packet[0], expected.loss, rtol=2e-12)
        np.testing.assert_allclose(ws.pw.get(stream=ws.stream), -project_unit_w_gradient(theta, expected.grad), rtol=2e-10, atol=2e-13)
        for side, fitted in enumerate((expected.left_model, expected.right_model)):
            np.testing.assert_allclose(ws.plans[side].coefficients.get(stream=ws.stream), fitted.coef_, rtol=2e-10, atol=2e-13)
            assert packet[4 + side] == pytest.approx(fitted.solve_result_.ridge_effective)


@pytest.mark.parametrize("batch,refresh,parallel", [(8, 32, True), (1, 1, False), (4, 3, True)])
@pytest.mark.parametrize("ridge", [0., 1e-5])
def test_optimizer_matches_cpu_and_serializes_with_cholesky_metadata(gpu, tmp_path, batch, refresh, parallel, ridge):
    import joblib

    rng = np.random.default_rng(404)
    X = rng.uniform(-1, 1, (160, 3))
    y = np.where(X[:, 0] > .1, X[:, 1] + 2., .5 * X[:, 2] - 1.)
    cpu = SoftObliqueSplitter(PolynomialRidgeModel(ridge=ridge, solver="normal"), .4, max_iters=6, random_state=4, alpha0=.2)
    expected = cpu.split(X, y)
    timing = SplitterTimingProfile()
    actual = replace(cpu, optimizer="cupy_cholesky", backtrack_batch_size=batch,
                     Xw_refresh=refresh, qr_parallel=parallel, split_timing_profile=timing).split(X, y)
    np.testing.assert_allclose(actual.w, expected.w, atol=1e-9)
    np.testing.assert_allclose(actual.z, expected.z, atol=1e-9)
    np.testing.assert_allclose(actual.predict(X), expected.predict(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.metadata["soft_loss_history"], expected.metadata["soft_loss_history"], rtol=2e-9)
    assert actual.metadata["step_size_history"] == expected.metadata["step_size_history"]
    assert actual.metadata["backtrack_history"] == expected.metadata["backtrack_history"]
    assert actual.metadata["physical_line_search_candidates"] == timing.line_search_candidate_count
    assert timing.least_squares_count == actual.n_iters + 2
    assert actual.metadata["optimizer"] == "cupy_cholesky"
    assert actual.metadata["complementary_grams"]
    assert actual.metadata["gram_floor_correction"]
    assert not actual.metadata["solver_fallback"]
    assert not actual.metadata["condition_estimation"]
    assert "qr_parallel" not in actual.metadata
    for model in (actual.left_model, actual.right_model):
        assert model.solve_result_.solver_requested == "normal"
        assert model.solve_result_.solver_used == "cholesky"
        assert model.solve_result_.cond_estimate is None
    path = tmp_path / "cholesky.joblib"
    joblib.dump(actual, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X), actual.predict(X))


def test_workspace_reuses_total_gram_across_complete_splits(gpu):
    rng = np.random.default_rng(406)
    X = rng.uniform(-1, 1, (120, 3))
    y = np.where(X[:, 0] > 0., X[:, 1] + 2., X[:, 2] - 1.)
    splitter = SoftObliqueSplitter(AffineRidgeModel(ridge=1e-4), .5,
                                   optimizer="cupy_cholesky", max_iters=3)
    ws = make_cholesky_workspace(splitter, X, y)
    ws.initialize(None, None)
    total, plans = ws.gram_pair.total, tuple(ws.plans)
    for seed in (4, 5):
        result = replace(splitter, random_state=seed).split(X, y, cholesky_workspace=ws)
        expected = replace(splitter, optimizer="cpu", random_state=seed).split(X, y)
        np.testing.assert_allclose(result.predict(X), expected.predict(X), rtol=1e-9, atol=1e-9)
        assert ws.gram_pair.total is total
        assert tuple(ws.plans) == plans


def test_singular_fit_fails_without_qr_or_cpu_fallback(gpu, monkeypatch):
    import art.models as models
    from art._cuda_qr import QrPlan

    rng = np.random.default_rng(405)
    X, y = rng.normal(size=(50, 2)), rng.normal(size=50)
    X[:, 1] = 0.  # An exactly zero Gram diagonal guarantees factorization failure.
    parent = AffineRidgeModel(solver="qr").fit(X, y)
    splitter = SoftObliqueSplitter(AffineRidgeModel(ridge=0.), 1., optimizer="cupy_cholesky")

    def unexpected(*args, **kwargs):
        raise AssertionError("Cholesky mode must not invoke another solver")

    monkeypatch.setattr(QrPlan, "solve", unexpected)
    monkeypatch.setattr(models, "solve_weighted_ridge", unexpected)
    with pytest.raises(SplitNotFoundError) as caught:
        splitter.split(X, y, parent_model=parent)
    assert caught.value.reason == "cholesky_failure"
