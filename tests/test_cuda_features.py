"""Feature order/layout parity with sklearn, on host kernels and actual CUDA."""

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from cuda_cpu_backend import cuda_backend  # noqa: F401
from art._cuda_batched_cholesky import BatchedCholeskyWorkspace
from art._cuda_features import FeaturePlan, build_features
from art.models import solve_weighted_ridge


@pytest.mark.parametrize("K,n,d,degree,bias,last", [
    (2, 7, 3, 1, True, True),   # Affine: constant last.
    (3, 5, 2, 1, True, False),  # Polynomial degree 1: constant first.
    (2, 5, 3, 1, False, False),
    (2, 7, 1, 2, False, False),
    (20, 3, 5, 2, True, False),
    (3, 3, 10, 2, False, False),
    (2, 2, 20, 2, True, False),
    (1, 2, 40, 2, True, False),
    (2, 259, 2, 2, True, False),  # Cross blocks, sample rows and node slots.
])
def test_features_match_cpu_basis_and_layout(cuda_backend, K, n, d, degree, bias, last):
    cp = cuda_backend
    X = np.random.default_rng(901).normal(size=(K, n, d))
    X[:, 0] = 0.  # Bias must not read X, including at the start of a node.
    x = cp.asarray(X, order="C")
    ptr = x.data.ptr
    phi = build_features(x, degree=degree, include_bias=bias, bias_last=last)
    cp.cuda.get_current_stream().synchronize()
    expected = PolynomialFeatures(degree=degree, include_bias=bias).fit_transform(X.reshape(-1, d))
    if last:
        expected = np.concatenate((expected[:, 1:], expected[:, :1]), axis=1)
    assert phi.shape == (K, n, expected.shape[1])
    assert phi.dtype == cp.float64 and phi.flags.c_contiguous
    np.testing.assert_array_equal(phi.get(), expected.reshape(phi.shape))
    np.testing.assert_array_equal(x.get(), X)
    assert x.shape == (K, n, d) and x.data.ptr == ptr


def test_quadratic_columns_have_one_copy_of_each_product(cuda_backend):
    cp = cuda_backend
    X = cp.asarray(np.array([[[2., 3.], [5., 7.]], [[11., 13.], [17., 19.]]]))
    phi = build_features(X)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(phi.get(), [
        [[1., 2., 3., 4., 6., 9.], [1., 5., 7., 25., 35., 49.]],
        [[1., 11., 13., 121., 143., 169.], [1., 17., 19., 289., 323., 361.]],
    ])


def test_feature_plan_reuses_indices_and_supplied_output(cuda_backend):
    cp, rng = cuda_backend, np.random.default_rng(903)
    plan = FeaturePlan(3, degree=2, include_bias=False)
    x = cp.empty((5, 7, 3), dtype=cp.float64)
    phi = cp.empty((5, 7, plan.p), dtype=cp.float64)
    addresses = [a.data.ptr for a in (plan.a, plan.b, phi)]
    for k in (5, 2, 1):
        X = rng.normal(size=(k, 7, 3))
        x[:k].set(X)
        phi.fill(-91.)
        out = phi[:k]
        assert plan.build(x[:k], out=out) is out
        cp.cuda.get_current_stream().synchronize()
        expected = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X.reshape(-1, 3))
        np.testing.assert_array_equal(phi.get()[:k], expected.reshape(k, 7, -1))
        np.testing.assert_array_equal(phi.get()[k:], -91.)
        assert addresses == [a.data.ptr for a in (plan.a, plan.b, phi)]
    with pytest.raises(ValueError, match="out"):
        plan.build(x, out=phi[:, :, :-1])


def test_device_design_is_borrowed_and_fits_match_cpu(cuda_backend):
    cp, rng = cuda_backend, np.random.default_rng(902)
    X, y = rng.normal(size=(3, 31, 3)), rng.normal(size=(3, 31))
    gates = rng.uniform(size=y.shape)
    x, target = cp.asarray(X), cp.asarray(y)
    phi = build_features(x)
    # Enqueue the fit on the same stream; no host feature transfer or D2D copy.
    with BatchedCholeskyWorkspace(phi, target, ridge=.01, copy=False) as ws:
        assert ws.phi is phi and ws.y is target
        ws.fit(cp.asarray(gates), nodes=[2, 0])
        ws.synchronize()
        actual = ws.coefficients.get()
        expected_phi = PolynomialFeatures(degree=2).fit_transform(X.reshape(-1, 3)).reshape(3, 31, -1)
        for j in (2, 0):
            for side, q in enumerate((1. - gates[j], gates[j])):
                ref = solve_weighted_ridge(expected_phi[j], y[j], q, .01, solver="normal")
                np.testing.assert_allclose(actual[j, side], ref.coefficients, rtol=3e-11, atol=1e-12)
    # Default construction continues to own independent copies of device inputs.
    with BatchedCholeskyWorkspace(phi, target) as ws:
        assert ws.phi.data.ptr != phi.data.ptr and ws.y.data.ptr != target.data.ptr


@pytest.mark.parametrize("bad", ["host", "dtype", "strides"])
def test_borrowed_data_requires_device_float64_contiguous_inputs(cuda_backend, bad):
    cp = cuda_backend
    phi, y = cp.asarray(np.ones((2, 5, 3))), cp.asarray(np.ones((2, 5)))
    if bad == "host": phi = np.ones(phi.shape)
    if bad == "dtype": y = cp.asarray(np.ones(y.shape, dtype=np.float32))
    if bad == "strides": phi = phi[:, :, ::-1]
    with pytest.raises(ValueError, match="copy=False"):
        BatchedCholeskyWorkspace(phi, y, copy=False)


@pytest.mark.parametrize("bad", ["dtype", "strides", "degree", "bias_last", "shape"])
def test_feature_builder_rejects_unsupported_inputs(cuda_backend, bad):
    cp = cuda_backend
    x, kwargs = cp.asarray(np.ones((2, 5, 3))), {}
    if bad == "dtype": x = cp.asarray(np.ones(x.shape, dtype=np.float32))
    if bad == "strides": x = x[:, :, ::-1]
    if bad == "degree": kwargs["degree"] = 3
    if bad == "bias_last": kwargs["bias_last"] = True
    if bad == "shape": x = x[0]
    with pytest.raises(ValueError):
        build_features(x, **kwargs)
