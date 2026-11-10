"""Run the same batched-kernel checks on host C++ and, when available, CUDA."""

import numpy as np
import pytest

from test_cuda_optimizer_kernels import compile_kernel_source


def gpu_module():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if not count:
        pytest.skip("CUDA device unavailable")
    return cp


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def kernel(request, tmp_path_factory):
    if request.param == "cpu":
        return compile_kernel_source(tmp_path_factory, "_cuda_optimizer_kernels.cu")
    cp = gpu_module()
    from art._cuda_optimizer_kernels import kernels

    functions = kernels(cp.cuda.Device().id)

    def run(name, grid, *args):
        # Transfers/synchronization are for assertions, not a runtime driver.
        device = [cp.array(a, order="K") if isinstance(a, np.ndarray) else
                  np.int32(a) if isinstance(a, (int, np.integer)) else np.float64(a)
                  for a in args]
        functions[name](grid, (256,), tuple(device))
        cp.cuda.get_current_stream().synchronize()
        for original, result in zip(args, device):
            if isinstance(original, np.ndarray):
                original[...] = cp.asnumpy(result)

    return run


def geometry(theta, p):
    w, pw = theta[:, :-1], p[:, :-1]
    return np.column_stack(((w * w).sum(1), (w * pw).sum(1),
                            (pw * pw).sum(1), p[:, -1]))


@pytest.mark.parametrize("advance,hard", [(0, 0), (1, 0), (0, 1)])
def test_gates_use_per_node_parameters_and_preserve_inactive_outputs(kernel, advance, hard):
    rng = np.random.default_rng(701)
    K, n, d = 3, 259, 4
    theta, v, u = rng.normal(size=(K, d + 1)), rng.normal(size=(K, n)), rng.normal(size=(K, n))
    u[:, :2] = [-1000., 1000.]
    before, pi = u.copy(), np.full((K, n), -7.)
    active = np.array([1, 0, 1], dtype=np.int32)
    temperature, alpha, inv_r = np.array([.3, .7, 2.]), np.array([.2, .4, .8]), np.array([.8, .9, .6])
    kernel("batched_gates", (2, K), n, d, advance, hard, active, temperature,
           theta, alpha, v, inv_r, u, pi)
    for j in range(K):
        if not active[j]:
            np.testing.assert_array_equal(u[j], before[j])
            np.testing.assert_array_equal(pi[j], -7.)
            continue
        score = (before[j] + alpha[j] * v[j]) * inv_r[j] if advance else before[j]
        expected = (score >= theta[j, -1]).astype(float) if hard else 1 / (
            1 + np.exp(-np.clip((score - theta[j, -1]) / temperature[j], -50, 50)))
        np.testing.assert_allclose(u[j], score, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(pi[j], expected, atol=3e-16)


def test_invalid_gate_inputs_are_exposed_to_the_fitter(kernel):
    K, n, d = 3, 2, 1
    u, pi = np.ones((K, n)), np.zeros((K, n))
    u[0, 0] = np.nan
    kernel("batched_gates", (1, K), n, d, 0, 0, np.ones(K, dtype=np.int32),
           np.array([1., 0., np.inf]), np.zeros((K, d + 1)), np.zeros(K),
           np.zeros_like(u), np.ones(K), u, pi)
    assert np.isnan(pi[0, 0]) and np.isfinite(pi[0, 1])
    assert np.isnan(pi[1:]).all()


def test_residuals_gradient_direction_and_side_major_metadata(kernel):
    rng = np.random.default_rng(702)
    K, n, d, M = 3, 259, 301, 2
    X, y, pred = rng.normal(size=(K, n, d)), rng.normal(size=(K, n)), rng.normal(size=(K, n, 2))
    theta, pi = rng.normal(size=(K, d + 1)), rng.uniform(size=(K, n))
    temperature, active = np.array([.3, .8, 1.7]), np.array([1, 0, 1], dtype=np.int32)
    errors, coeff, partials = np.full_like(pred, -7.), np.full_like(y, -7.), np.full((K, 3, M), -7.)
    kernel("batched_residual_stats", (M, K), n, active, temperature, y, pred, pi,
           errors, coeff, partials)
    totals = partials.sum(axis=2)
    gw = (X.swapaxes(-1, -2) @ coeff[..., None])[..., 0]
    p, geo, packet = np.full_like(theta, -7.), np.full((K, 4), -7.), np.full((K, 7), -7.)
    ridge = np.array([[.1, .2, .3], [.4, .5, .6]])  # physical (2,K), not (K,2)
    status = np.zeros((2, K), dtype=np.int32)
    status[1, 2] = 3
    kernel("batched_direction", (K,), K, d, active, theta, gw, totals,
           ridge, status, p, geo, packet)
    for j in (0, 2):
        e = (y[j, :, None] - pred[j])**2
        c = (e[:, 1] - e[:, 0]) * pi[j] * (1 - pi[j]) / (n * temperature[j])
        g = np.r_[X[j].T @ c, -c.sum()]
        w = theta[j, :-1]
        expected_p = np.r_[-g[:-1] + (w @ g[:-1]) / (w @ w) * w, -g[-1]]
        loss = np.mean((1 - pi[j]) * e[:, 0] + pi[j] * e[:, 1])
        np.testing.assert_allclose(errors[j], e, atol=1e-14)
        np.testing.assert_allclose(coeff[j], c, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(p[j], expected_p, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(geo[j], geometry(theta[j:j+1], expected_p[None])[0], atol=1e-12)
        np.testing.assert_allclose(packet[j], [loss, np.linalg.norm(expected_p), g @ expected_p,
                                              np.count_nonzero(pi[j] >= .5),
                                              ridge[0, j], ridge[1, j], j == 2], atol=1e-12)
    for array in (errors, coeff, partials, p, geo, packet):
        np.testing.assert_array_equal(array[1], -7.)


@pytest.mark.parametrize("K,n,H,S,offset", [(1, 1, 1, 1, 0), (3, 259, 8, 11, 0), (3, 259, 8, 11, 8)])
def test_trials_and_reduction_match_explicit_projection(kernel, K, n, H, S, offset):
    rng = np.random.default_rng(703)
    d, M = 3, (n + 255) // 256
    X, theta, p = rng.normal(size=(K, n, d)), rng.normal(size=(K, d + 1)), rng.normal(size=(K, d + 1))
    u, v = (X @ theta[:, :-1, None])[..., 0], (X @ p[:, :-1, None])[..., 0]
    errors, temperature = rng.uniform(size=(K, n, 2)), np.linspace(.3, 1.7, K)
    alphas = (np.arange(K) + 1.)[:, None] * .5**np.arange(S)[None, :]
    searching = np.ones(K, dtype=np.int32)
    if K > 1: searching[1] = 0
    partials, losses = np.full((K, H, M), np.nan), np.full((K, H), -7.)
    kernel("batched_trial_losses", (M, H, K), n, d, S, offset, searching, temperature,
           theta, u, v, errors, geometry(theta, p), alphas, partials)
    kernel("batched_trial_reduce", (H, K), M, searching, partials, losses)
    for j in range(K):
        if not searching[j]:
            assert np.isnan(partials[j]).all()
            np.testing.assert_array_equal(losses[j], -7.)
            continue
        for k in range(H):
            if offset + k >= S:
                assert np.isposinf(partials[j, k]).all() and np.isposinf(losses[j, k])
                continue
            candidate = theta[j] + alphas[j, offset + k] * p[j]
            candidate[:-1] /= np.linalg.norm(candidate[:-1])
            pi = 1 / (1 + np.exp(-np.clip((X[j] @ candidate[:-1] - candidate[-1]) / temperature[j], -50, 50)))
            expected = np.mean((1 - pi) * errors[j, :, 0] + pi * errors[j, :, 1])
            np.testing.assert_allclose(losses[j, k], expected, rtol=1e-13, atol=1e-15)


def test_selection_freezes_early_accepts_and_exhausts_only_pending_nodes(kernel):
    K, H, S = 4, 8, 11
    alphas = (np.arange(K) + 1.)[:, None] * .5**np.arange(S)[None, :]
    packet = np.tile([10., 1., -1., 0., 0., 0., 0.], (K, 1))
    searching = np.array([1, 1, 1, 0], dtype=np.int32)
    selected = np.array([-1, -1, -1, 4], dtype=np.int32)
    value, step = np.full(K, -7.), np.full(K, -7.)
    losses = np.full((K, H), 12.)
    losses[0, [2, 4]] = [8., 1.]  # Earliest passing candidate wins, not minimum loss.
    kernel("batched_select_trial", (K,), S, H, 0, .1, losses, alphas,
           packet, searching, selected, value, step)
    np.testing.assert_array_equal(searching, [0, 1, 1, 0])
    np.testing.assert_array_equal(selected, [2, -1, -1, 4])
    first_value, first_step = value[0], step[0]
    losses[0] = np.nan  # Must never be read again.
    losses[1, 1] = 8.
    losses[2, 3:] = 0.  # Would pass, but lies beyond the candidate budget.
    kernel("batched_select_trial", (K,), S, H, 8, .1, losses, alphas,
           packet, searching, selected, value, step)
    np.testing.assert_array_equal(searching, 0)
    np.testing.assert_array_equal(selected, [2, 9, -1, 4])
    np.testing.assert_allclose(value, [first_value, 8., 12., -7.])
    np.testing.assert_allclose(step, [first_step, alphas[1, 9], alphas[2, 10], -7.])


@pytest.mark.parametrize("problem", ["zero_norm", "nan_norm", "nan_score", "zero_temperature", "inf_temperature"])
def test_invalid_trials_are_rejected_without_affecting_other_nodes(kernel, problem):
    K, n, d, H, M = 3, 259, 1, 2, 2
    theta, u, v = np.zeros((K, d + 1)), np.zeros((K, n)), np.zeros((K, n))
    geo, temperature = np.tile([1., 0., 0., 0.], (K, 1)), np.ones(K)
    if problem == "zero_norm": geo[1, 0] = 0.
    if problem == "nan_norm": geo[1, 0] = np.nan
    if problem == "nan_score": u[1, 0] = np.nan
    if problem == "zero_temperature": temperature[1] = 0.
    if problem == "inf_temperature": temperature[1] = np.inf
    searching = np.array([1, 1, 0], dtype=np.int32)
    partials, losses = np.full((K, H, M), -7.), np.full((K, H), -7.)
    kernel("batched_trial_losses", (M, H, K), n, d, H, 0, searching, temperature,
           theta, u, v, np.ones((K, n, 2)), geo, np.ones((K, H)), partials)
    kernel("batched_trial_reduce", (H, K), M, searching, partials, losses)
    np.testing.assert_allclose(losses[0], 1.)
    assert np.isposinf(losses[1]).all()
    np.testing.assert_array_equal(losses[2], -7.)
    selected = np.array([-1, -1, 7], dtype=np.int32)
    packet = np.tile([10., 1., -1., 0., 0., 0., 0.], (K, 1))
    kernel("batched_select_trial", (K,), H, H, 0, .1, losses, np.ones((K, H)),
           packet, searching, selected, np.full(K, -7.), np.full(K, -7.))
    np.testing.assert_array_equal(searching, 0)
    np.testing.assert_array_equal(selected, [0, -1, 7])


def test_selection_rejects_failed_current_point(kernel):
    searching, selected = np.ones(2, dtype=np.int32), np.full(2, -1, dtype=np.int32)
    packet = np.tile([10., 1., -1., 0., 0., 0., 0.], (2, 1))
    packet[1, 6] = 1.
    value, step = np.full(2, -7.), np.full(2, -7.)
    kernel("batched_select_trial", (2,), 1, 1, 0, .1, np.ones((2, 1)), np.ones((2, 1)),
           packet, searching, selected, value, step)
    np.testing.assert_array_equal(searching, 0)
    np.testing.assert_array_equal(selected, [0, -1])
    np.testing.assert_array_equal(value, [1., -7.])


def test_commit_and_cached_score_advance_use_each_accepted_step(kernel):
    rng = np.random.default_rng(704)
    K, n, d = 3, 259, 4
    X, theta, p = rng.normal(size=(K, n, d)), rng.normal(size=(K, d + 1)), rng.normal(size=(K, d + 1))
    original = theta.copy()
    u, v = (X @ theta[:, :-1, None])[..., 0], (X @ p[:, :-1, None])[..., 0]
    before_u = u.copy()
    selected, step = np.array([0, -1, 9], dtype=np.int32), np.array([.2, np.nan, .03])
    geo, inv_r = geometry(theta, p), np.full(K, -7.)
    geo[1] = np.nan  # Skipped before any shared-memory barriers or invalid arithmetic.
    kernel("batched_commit_theta", (K,), d, selected, step, p, geo, theta, inv_r)
    active, pi = (selected >= 0).astype(np.int32), np.full((K, n), -7.)
    kernel("batched_gates", (2, K), n, d, 1, 0, active, np.ones(K),
           theta, step, v, inv_r, u, pi)
    for j in (0, 2):
        expected = original[j] + step[j] * p[j]
        norm = np.linalg.norm(expected[:-1])
        expected[:-1] /= norm
        np.testing.assert_allclose(theta[j], expected, atol=1e-14)
        np.testing.assert_allclose(inv_r[j], 1 / norm, rtol=1e-14)
        np.testing.assert_allclose(u[j], X[j] @ expected[:-1], atol=1e-14)
    np.testing.assert_array_equal(theta[1], original[1])
    np.testing.assert_array_equal(u[1], before_u[1])
    np.testing.assert_array_equal(pi[1], -7.)
    assert inv_r[1] == -7.


def test_cuda_current_point_connects_to_batched_cholesky():
    cp = gpu_module()
    from art._cuda_batched_cholesky import BatchedCholeskyWorkspace
    from art._cuda_optimizer_kernels import kernels
    from art.models import AffineRidgeModel
    from art.objectives import SoftObliqueRidgeObjective

    rng = np.random.default_rng(705)
    K, n, d = 3, 41, 3
    X, y, theta = rng.normal(size=(K, n, d)), rng.normal(size=(K, n)), rng.normal(size=(K, d + 1))
    theta[:, :-1] /= np.linalg.norm(theta[:, :-1], axis=1)[:, None]
    phi = np.concatenate((X, np.ones((K, n, 1))), axis=2)
    temperature = np.array([.3, .8, 1.7])
    functions = kernels(cp.cuda.Device().id)

    def call(name, grid, *args):
        functions[name](grid, (256,), tuple(np.int32(a) if isinstance(a, int) else a for a in args))

    with BatchedCholeskyWorkspace(phi, y, ridge=1e-4, weight_floor=.07) as ws:
        x, th, temp = cp.asarray(X), cp.asarray(theta), cp.asarray(temperature)
        active, alpha, inv_r = cp.ones(K, dtype=cp.int32), cp.zeros(K), cp.ones(K)
        u = (x @ th[:, :-1, None])[..., 0]
        pi, v = cp.empty((K, n)), cp.zeros((K, n))
        call("batched_gates", (1, K), n, d, 0, 0, active, temp, th, alpha, v, inv_r, u, pi)
        beta, _ = ws.fit(pi)
        pred = cp.ascontiguousarray(ws.phi @ beta.swapaxes(-1, -2))
        errors, coeff, partials = cp.empty_like(pred), cp.empty((K, n)), cp.empty((K, 3, 1))
        call("batched_residual_stats", (1, K), n, active, temp, ws.y, pred, pi, errors, coeff, partials)
        gw = (x.swapaxes(-1, -2) @ coeff[..., None])[..., 0]
        totals, p, geo, packet = partials.sum(2), cp.empty_like(th), cp.empty((K, 4)), cp.empty((K, 7))
        call("batched_direction", (K,), K, d, active, th, gw, totals, ws._ridge, ws._status, p, geo, packet)
        ws.synchronize()
        actual_beta, actual_p, actual_packet = cp.asnumpy(beta), cp.asnumpy(p), cp.asnumpy(packet)
        for j in range(K):
            ref = SoftObliqueRidgeObjective(temperature[j], AffineRidgeModel(1e-4, solver="normal"),
                                            X[j], y[j], weight_floor=.07).value_and_grad(theta[j])
            expected_p = -ref.grad.copy()
            expected_p[:-1] += (theta[j, :-1] @ ref.grad[:-1]) * theta[j, :-1]
            np.testing.assert_allclose(actual_beta[j, 0], ref.left_model.coef_, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(actual_beta[j, 1], ref.right_model.coef_, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(actual_p[j], expected_p, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(actual_packet[j, :3], [ref.loss, np.linalg.norm(expected_p),
                                                             ref.grad @ expected_p], rtol=1e-11, atol=1e-13)
            assert actual_packet[j, 6] == 0.
