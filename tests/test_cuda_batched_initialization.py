"""GPU initialization formulas, default dispatch, and persistent RNG behavior."""

import numpy as np
import pytest

from cuda_cpu_backend import cuda_backend  # noqa: F401
from art._cuda_batched_optimizer import BatchedOptimizerWorkspace
from art.models import AffineRidgeModel
from art.splitters import SoftObliqueSplitter, SplitNotFoundError


def problem(n=13, d=3):
    rng = np.random.default_rng(921)
    X = rng.normal(size=(4, n, d))
    X[:, 0, 0], X[:, 1, 0] = -1., 1.
    y = np.where(X[:, :, 0] >= 0., 1. + X[:, :, -1], -.5 * X[:, :, -1])
    return X, y


@pytest.mark.parametrize("n,d", [(12, 1), (13, 3), (13, 259)])
def test_gpu_start_matches_numpy_normalization_median_and_preserves_scores(cuda_backend, monkeypatch, n, d):
    X, y = problem(n, d)
    s = SoftObliqueSplitter(AffineRidgeModel(), .7, random_state=71)
    with BatchedOptimizerWorkspace.allocate(s, 5, n, d) as ws:
        ws.load(X, y)
        ws.theta.fill(-91.); ws.u.fill(-91.)
        mask = np.array([1, 0, 1, 1, 0], dtype=bool)
        raw = np.random.default_rng(922).normal(size=(3, d))
        raw[1] = 1e-20 if d == 3 else 0.  # Tiny/zero norm matches the CPU fallback.
        draws, products = [], []

        def normal(handle, ptr, size, mean, std):
            assert mean == 0. and std == 1.
            draws.append(size)
            values = np.zeros(size)
            values[:raw.size] = raw.ravel()
            ws._normal[:size].set(values, stream=ws.stream)

        original = ws.products["u"].run

        def multiply():
            products.append(1)
            original()

        monkeypatch.setattr(ws.curand, "generateNormalDouble", normal)
        monkeypatch.setattr(ws.products["u"], "run", multiply)
        normal_ptr = ws._normal.data.ptr
        ws.start(None, mask)
        ws.stream.synchronize()
        theta, u, pi = ws.theta.get(), ws.u.get(), ws.pi.get()
        assert products == [1]  # The first gate evaluation reuses this Xw.
        assert draws == [2 * ((raw.size + 1) // 2)]
        assert normal_ptr == ws._normal.data.ptr
        for row, j in enumerate(ws.order[:ws.count]):
            norm = np.linalg.norm(raw[row])
            w = raw[row].copy()
            if norm <= 1e-12:
                w[0] = 1.
                norm = np.linalg.norm(w)
            w /= norm
            scores = X[j] @ w
            z = np.median(scores)
            np.testing.assert_allclose(theta[j, :-1], w, rtol=1e-13, atol=1e-14)
            if row == 1:
                np.testing.assert_array_equal(theta[j, :-1], w)
            np.testing.assert_allclose(u[j], scores, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(theta[j, -1], z, rtol=1e-12, atol=1e-12)
            expected_pi = 1. / (1. + np.exp(-np.clip((scores - z) / .7, -50, 50)))
            np.testing.assert_allclose(pi[j], expected_pi, rtol=1e-12, atol=1e-12)
            assert min(np.count_nonzero(u[j] < theta[j, -1]),
                       np.count_nonzero(u[j] >= theta[j, -1])) == n // 2
        np.testing.assert_array_equal(theta[~mask], -91.)
        np.testing.assert_array_equal(u[~mask], -91.)


def test_gpu_rng_is_seeded_once_and_advances_across_starts_and_loads(cuda_backend, monkeypatch):
    X, y = problem()
    s = SoftObliqueSplitter(AffineRidgeModel(), .7, random_state=72)

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU initialization must not use the CPU initializer")

    monkeypatch.setattr(s, "_initialize_theta", forbidden)
    mask = np.array([1, 0, 1, 1, 0], dtype=bool)
    with BatchedOptimizerWorkspace.allocate(s, 5, 13, 3) as ws:
        ws.load(X, y)
        handle = ws.rng_handle
        ws.start(None, mask)
        first = ws.theta.get(stream=ws.stream)[mask].copy()
        ws.start(None, mask)
        second = ws.theta.get(stream=ws.stream)[mask].copy()
        ws.load(X, y)
        ws.start(None, mask)
        third = ws.theta.get(stream=ws.stream)[mask].copy()
        assert handle == ws.rng_handle
        assert not np.allclose(first, second) and not np.allclose(first, third)
        with BatchedOptimizerWorkspace.allocate(s, 5, 13, 3) as same:
            assert same.seed == ws.seed
            same.load(X, y)
            same.start(None, mask)
            np.testing.assert_allclose(same.theta.get(stream=same.stream)[mask], first,
                                       rtol=1e-13, atol=1e-14)
    assert ws.rng_handle is None


def test_gpu_initialization_uses_nondefault_stream(cuda_backend):
    cp = cuda_backend
    if not hasattr(cp.cuda, "Stream"):
        pytest.skip("Non-default stream behavior requires actual CUDA")
    X, y = problem()
    s = SoftObliqueSplitter(AffineRidgeModel(ridge=1e-4), .7, max_iters=0, random_state=74)
    stream = cp.cuda.Stream(non_blocking=True)
    with BatchedOptimizerWorkspace.allocate(s, 5, 13, 3, stream=stream) as ws:
        ws.load(X, y)
        ws.start(None, np.array([1, 1, 1, 1, 0], dtype=bool))
        theta = ws.theta.get(stream=stream)[:len(X)]
        np.testing.assert_allclose(np.linalg.norm(theta[:, :-1], axis=1), 1., atol=1e-14)
        scores = (X @ theta[:, :-1, None])[..., 0]
        np.testing.assert_allclose(theta[:, -1], np.median(scores, axis=1), rtol=1e-12, atol=1e-12)
        results = ws.optimize(parent_loss=np.ones(len(X)))
        assert all(not isinstance(result, SplitNotFoundError) for result in results), results


@pytest.mark.parametrize("provided,expected_draws", [("none", 3), ("first", 2), ("all", 0)])
def test_default_and_explicit_restart_dispatch(cuda_backend, monkeypatch, provided, expected_draws):
    X, y = problem()
    s = SoftObliqueSplitter(AffineRidgeModel(ridge=1e-4), .7, max_iters=0,
                            min_side_points=1, n_restarts=3, random_state=73)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU random initial-state construction is disabled")

    monkeypatch.setattr(s, "_initialize_theta", forbidden)
    with BatchedOptimizerWorkspace.allocate(s, 5, 13, 3) as ws:
        ws.load(X, y)
        original, draws, starts = ws.curand.generateNormalDouble, [], []

        def generate(*args):
            draws.append(args[2])
            return original(*args)

        original_start = ws.start

        def start(theta, mask):
            starts.append(theta is None)
            original_start(theta, mask)

        monkeypatch.setattr(ws.curand, "generateNormalDouble", generate)
        monkeypatch.setattr(ws, "start", start)
        theta = np.tile([1., 0., 0., 0.], (len(X), 1))
        initial = None if provided == "none" else theta if provided == "first" else np.stack([theta] * 3)
        results = ws.optimize(initial, parent_loss=np.ones(len(X)))
        assert len(draws) == expected_draws and sum(starts) == expected_draws
        assert all(size % 2 == 0 for size in draws)
        for result in results:
            assert not isinstance(result, SplitNotFoundError), result
            np.testing.assert_allclose(np.linalg.norm(result.w), 1., atol=1e-14)
            assert np.isfinite(result.loss)
            if provided == "none":
                assert result.metadata["initialization"] == "gpu"
                assert result.metadata["initialization_seed"] == ws.seed
            if provided == "all":
                assert result.metadata["initialization"] == "provided"
                assert result.metadata["initialization_seed"] is None
