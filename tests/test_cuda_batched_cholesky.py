"""Check batched arithmetic on CPU and native library calls on real CUDA."""

import builtins
import numpy as np
import pytest

from art._cuda_batched_cholesky import BatchedCholeskyWorkspace
from art.models import solve_weighted_ridge
from test_cuda_optimizer_kernels import compile_kernel_source


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    return compile_kernel_source(tmp_path_factory, "_cuda_batched_cholesky_kernels.cu")


def cpu_workspace(kernel, phi, y, ridge=0.0, floor=1e-12):
    """Run the production assembly/kernels with NumPy; no CUDA emulation."""
    ws = BatchedCholeskyWorkspace.__new__(BatchedCholeskyWorkspace)
    ws.handle = None
    ws.cp, ws.phi, ws.y = np, phi, y
    ws.K, ws.n, ws.p = phi.shape
    ws.ridge, ws.weight_floor = ridge, floor
    ws._phi_t = phi.swapaxes(-1, -2)
    ws.weights = np.empty((2, ws.K, ws.n))
    ws._weighted_phi, ws._weighted_y = np.empty_like(phi), np.empty_like(y)
    ws.grams = np.empty((2, ws.K, ws.p, ws.p))
    ws._factor_storage = np.empty_like(ws.grams)
    ws.factors = ws._factor_storage.swapaxes(-1, -2)
    ws._rhs = np.empty((2, ws.K, ws.p))
    ws._ridge = np.empty((2, ws.K))
    ws._status = np.empty((2, ws.K), dtype=np.int32)
    ws._invalid = np.empty(ws.K, dtype=np.int32)
    ws._factor_info = np.empty(2 * ws.K, dtype=np.int32)
    ws._solve_info = np.zeros(1, dtype=np.int32)
    names = ("batch_weights", "batch_regularize", "batch_protect", "batch_finish")
    ws.functions = {
        name: lambda grid, block, args, name=name: kernel(name, (grid[0], 1), *args)
        for name in names
    }
    return ws


@pytest.mark.parametrize("ridge", [0., .03])
@pytest.mark.parametrize("floor", [0., 1e-12, .1, .75])
def test_direct_assembly_and_column_major_packing(kernel, ridge, floor):
    rng = np.random.default_rng(601)
    phi, y = rng.normal(size=(3, 17, 4)), rng.normal(size=(3, 17))
    pi = rng.uniform(size=(3, 17))
    pi[:, :4] = [0., 1., 1e-20, 1. - 1e-15]
    ws = cpu_workspace(kernel, phi, y, ridge, floor)
    buffers = [ws.weights, ws.grams, ws._weighted_phi, ws._weighted_y, ws._factor_storage, ws._rhs]
    for gates in (pi, 1. - pi):
        ws._assemble(gates)
        for side in range(2):
            q = np.maximum(1. - gates if side == 0 else gates, floor)
            np.testing.assert_array_equal(ws.weights[side], q)
            for j in range(ws.K):
                gram = phi[j].T @ (q[j, :, None] * phi[j])
                lam = ridge * np.trace(gram) / ws.p
                np.testing.assert_allclose(ws.grams[side, j], gram, rtol=2e-15, atol=1e-14)
                np.testing.assert_allclose(ws.factors[side, j], gram + lam * np.eye(ws.p))
                np.testing.assert_allclose(ws._rhs[side, j], phi[j].T @ (q[j] * y[j]))
                assert ws._ridge[side, j] == pytest.approx(lam)
                assert ws.factors[side, j].flags.f_contiguous
        np.testing.assert_array_equal(ws._status, 0)
    assert all(a is b for a, b in zip(buffers, [ws.weights, ws.grams, ws._weighted_phi,
                                             ws._weighted_y, ws._factor_storage, ws._rhs]))


@pytest.mark.parametrize("p", [1, 259])
def test_trace_reduction_across_all_columns(kernel, p):
    rng = np.random.default_rng(602)
    phi, y = rng.normal(size=(1, 11, p)), rng.normal(size=(1, 11))
    ws = cpu_workspace(kernel, phi, y, ridge=.2)
    ws._assemble(np.full((1, 11), .3))
    expected = .2 * np.sum(phi**2) / p * np.array([.7, .3])
    np.testing.assert_allclose(ws._ridge[:, 0], expected, rtol=2e-15)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.1, 1.1])
def test_bad_gate_marks_only_its_node_and_resets_on_next_fit(kernel, value):
    ws = cpu_workspace(kernel, np.ones((3, 8, 1)), np.ones((3, 8)))
    pi = np.full((3, 8), .4)
    pi[1, 2] = value
    ws._assemble(pi)
    np.testing.assert_array_equal(ws._status, [[0, -1, 0], [0, -1, 0]])
    np.testing.assert_array_equal(ws.factors[:, 1], np.ones((2, 1, 1)))
    np.testing.assert_array_equal(ws._rhs[:, 1], 0)
    pi[1, 2] = .4
    ws._assemble(pi)
    np.testing.assert_array_equal(ws._status, 0)


def test_factor_failures_and_nonfinite_solves_are_isolated(kernel):
    p, count = 3, 6
    a = np.tile(np.eye(p), (count, 1, 1))
    b = np.ones((count, p))
    info = np.array([0, 2, 0, 0, 0, 0], dtype=np.int32)
    status = np.array([0, 0, 0, -1, 0, 0], dtype=np.int32)
    # Storage is column-major: only the lower triangle is meaningful after potrf.
    a[0, 0, 2] = 9.  # A nonsymmetric marker makes a transposition error visible.
    a[0, 2, 0] = np.nan  # Solver scratch in the upper triangle is ignored.
    a[2, 1, 1] = 0.
    kernel("batch_protect", (count, 1), p, a, b, info, status)
    np.testing.assert_array_equal(status, [0, 2, -2, -1, 0, 0])
    np.testing.assert_array_equal(a[1:4], np.tile(np.eye(p), (3, 1, 1)))
    np.testing.assert_array_equal(b[1:4], 0)
    assert a[0, 0, 2] == 9. and np.isnan(a[0, 2, 0])
    b[4, 1] = np.inf
    kernel("batch_finish", (count, 1), p, b, np.zeros(1, dtype=np.int32), status)
    np.testing.assert_array_equal(status, [0, 2, -2, -1, -3, 0])
    assert np.isnan(b[1:5]).all()
    np.testing.assert_array_equal(b[[0, 5]], 1.)
    kernel("batch_finish", (count, 1), p, b, np.array([-4], dtype=np.int32), status)
    np.testing.assert_array_equal(status, [-3, 2, -2, -1, -3, -3])
    assert np.isnan(b).all()


def test_nonfinite_assembly_is_flagged_before_factorization(kernel):
    ws = cpu_workspace(kernel, np.ones((2, 8, 2)), np.ones((2, 8)), ridge=.1)
    ws._assemble(np.full((2, 8), .4))
    ws.grams[1, 0, 0, 0] = np.inf
    ws.functions["batch_regularize"]((4,), (256,), (
        ws.K, ws.p, ws.ridge, ws.grams, ws._invalid, ws._factor_storage,
        ws._rhs, ws._ridge, ws._status,
    ))
    np.testing.assert_array_equal(ws._status, [[0, 0], [-2, 0]])
    np.testing.assert_array_equal(ws.factors[1, 0], np.eye(2))


@pytest.mark.parametrize("phi,y,kwargs,message", [
    (np.ones((2, 3)), np.ones(3), {}, "Phi"),
    (np.ones((0, 3, 2)), np.ones((0, 3)), {}, "Phi"),
    (np.ones((2, 3, 2)), np.ones((2, 3, 1)), {}, "y"),
    (np.ones((2, 3, 2)), np.ones((2, 3)), {"ridge": -1}, "ridge"),
    (np.ones((2, 3, 2)), np.ones((2, 3)), {"ridge": np.nan}, "ridge"),
    (np.ones((2, 3, 2)), np.ones((2, 3)), {"weight_floor": np.inf}, "weight_floor"),
    (np.ones((2, 3, 2)), np.ones((2, 3)), {"weight_floor": -1}, "weight_floor"),
    (np.ones((2, 3, 2), dtype=complex), np.ones((2, 3)), {}, "real"),
])
def test_validation_precedes_cupy_import(monkeypatch, phi, y, kwargs, message):
    original = builtins.__import__

    def guarded(name, *args, **kw):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("Validation must not require CuPy")
        return original(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match=message):
        BatchedCholeskyWorkspace(phi, y, **kwargs)


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if not count:
        pytest.skip("CUDA device unavailable")
    return cp


@pytest.mark.parametrize("K,p,ridge,floor", [
    (1, 1, 0., 0.), (1, 6, 0., 1e-12), (3, 6, .03, .07),
    (3, 6, 0., .75), (20, 21, 1e-5, 1e-12), (3, 231, 1e-5, .07),
    (1, 259, .03, 0.),
])
def test_native_batched_fits_match_cpu_and_reuse_storage(gpu, K, p, ridge, floor):
    cp, rng = gpu, np.random.default_rng(603)
    phi, y = rng.normal(size=(K, 5 * p + 4, p)), rng.normal(size=(K, 5 * p + 4))
    pi = rng.uniform(size=y.shape)
    pi[:, :4] = [0., 1., 1e-20, 1. - 1e-15]
    stream = cp.cuda.Stream(non_blocking=True)
    with BatchedCholeskyWorkspace(phi, y, ridge, floor, stream=stream) as ws:
        buffers = [ws.phi, ws.y, ws.weights, ws.grams, ws._factor_storage, ws._rhs]
        addresses = [a.data.ptr for a in buffers]
        for iteration, gates in enumerate((pi, 1. - pi)):
            with stream:
                device_pi = cp.asarray(gates)
            coefficients, status = ws.fit(device_pi)
            assert coefficients is ws.coefficients and status is ws.status
            ws.synchronize()
            np.testing.assert_array_equal(cp.asnumpy(status), 0)
            actual = cp.asnumpy(coefficients)
            if iteration == 0:
                first_result = actual.copy()
            for j in range(K):
                for side in range(2):
                    weights = 1. - gates[j] if side == 0 else gates[j]
                    ref = solve_weighted_ridge(phi[j], y[j], weights, ridge,
                                               weight_floor=floor, solver="normal")
                    np.testing.assert_allclose(actual[j, side], ref.coefficients,
                                               rtol=3e-11, atol=1e-12)
                    assert float(ws.ridge_effective[j, side]) == pytest.approx(ref.ridge_effective)
            assert addresses == [a.data.ptr for a in buffers]
        # Host gates use the same fit interface; fixed data are copied just once.
        phi.fill(0.); y.fill(0.)
        ws.fit(pi)
        ws.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(ws.status), 0)
        assert np.any(cp.asnumpy(ws.phi) != 0.)
        np.testing.assert_allclose(cp.asnumpy(ws.coefficients), first_result, rtol=2e-12, atol=1e-13)
    with pytest.raises(RuntimeError, match="closed"):
        ws.fit(pi)


def test_native_failure_isolation_and_recovery(gpu):
    cp, rng = gpu, np.random.default_rng(604)
    phi, y = rng.normal(size=(3, 20, 3)), rng.normal(size=(3, 20))
    phi[1, :, 2] = 0.  # Both fits at node 1 fail at pivot 3, deterministically.
    pi = np.full((3, 20), .4)
    pi[2] = 0.  # Only the right fit at node 2 is empty.
    with BatchedCholeskyWorkspace(phi, y, weight_floor=0.) as ws:
        coefficients, status = ws.fit(pi)
        ws.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(status), [[0, 0], [3, 3], [0, 1]])
        result = cp.asnumpy(coefficients)
        assert np.isfinite(result[0]).all() and np.isfinite(result[2, 0]).all()
        assert np.isnan(result[1]).all() and np.isnan(result[2, 1]).all()
        for side, q in enumerate((1. - pi[0], pi[0])):
            ref = solve_weighted_ridge(phi[0], y[0], q, 0., solver="normal")
            np.testing.assert_allclose(result[0, side], ref.coefficients, rtol=2e-12, atol=1e-13)
        pi[0, 0] = np.nan
        ws.fit(pi); ws.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(status)[0], [-1, -1])
        pi[0, 0] = .4
        pi[2] = .4
        ws.fit(pi); ws.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(status), [[0, 0], [3, 3], [0, 0]])


def test_gpu_input_validation(gpu):
    phi, y = np.ones((2, 8, 2)), np.ones((2, 8))
    phi[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        BatchedCholeskyWorkspace(phi, y)
    phi[0, 0, 0], y[0, 0] = 1., np.inf
    with pytest.raises(ValueError, match="finite"):
        BatchedCholeskyWorkspace(phi, y)
    y[0, 0] = 1.
    with BatchedCholeskyWorkspace(phi, y, ridge=.1) as ws:
        with pytest.raises(ValueError, match="shape"):
            ws.fit(np.ones(8))
        with pytest.raises(ValueError, match="real"):
            ws.fit(np.ones((2, 8), dtype=complex))
