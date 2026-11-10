"""Build resident linear/quadratic designs without a host feature matrix.

X stays C-contiguous (K,n,d); Phi is C-contiguous float64 (K,n,p), as required
by the batched Cholesky and prediction products. The basis is shared by nodes.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


def _columns(d, degree, include_bias, bias_last):
    """Return coordinate pairs in model coefficient order; -1 means no factor.

    Polynomial order is bias, linear terms, then x_i*x_j for i <= j, with i
    outermost. For d=2: [1, x0, x1, x0*x0, x0*x1, x1*x1]. Affine models
    instead use [x0, ..., x_(d-1), 1]. No cross term is repeated or scaled.
    """
    if not isinstance(degree, (int, np.integer)) or degree not in (1, 2):
        raise ValueError("GPU features support only degree 1 or 2.")
    if bias_last and degree != 1:
        raise ValueError("bias_last is reserved for affine features (degree 1).")
    pairs = [(i, -1) for i in range(d)]
    if degree == 2:
        pairs.extend((i, j) for i in range(d) for j in range(i, d))
    if include_bias:
        pairs.insert(len(pairs) if bias_last else 0, (-1, -1))
    return np.asarray(pairs, dtype=np.int32).T.copy()


@lru_cache(maxsize=None)
def _kernel(device_id):
    import cupy as cp

    return cp.RawModule(
        code=Path(__file__).with_suffix(".cu").read_text(),
        options=("--std=c++11",),
    ).get_function("polynomial_features")


class FeaturePlan:
    """Upload a fixed linear/quadratic basis once; fill supplied Phi buffers.

    Create/use on the same stream, or establish ordering before switching.
    Inputs and output must be C-contiguous float64 on the original device;
    no sample-sized scratch is allocated.
    """

    def __init__(self, d, *, degree=2, include_bias=True, bias_last=False):
        if not isinstance(d, (int, np.integer)) or not 0 < d <= np.iinfo(np.int32).max:
            raise ValueError("d must be a positive CUDA integer.")
        columns = _columns(d, degree, include_bias, bias_last)
        self.d, self.p = int(d), columns.shape[1]
        if self.p > np.iinfo(np.int32).max:
            raise ValueError("Feature dimensions exceed the CUDA integer limit.")
        import cupy as cp

        self.cp, self.device_id = cp, cp.cuda.Device().id
        self.a, self.b = cp.asarray(columns)
        self.kernel = _kernel(self.device_id)

    def build(self, X, out=None):
        cp = self.cp
        if X.ndim != 3 or min(X.shape) < 1 or X.shape[2] != self.d:
            raise ValueError("X must have nonempty shape (k,n,d) matching the basis.")
        if (not isinstance(X, cp.ndarray) or X.dtype != cp.float64
                or not X.flags.c_contiguous or X.device.id != self.device_id
                or cp.cuda.Device().id != self.device_id):
            raise ValueError("X must be C-contiguous float64 on the current CUDA device.")
        k, n, d = X.shape
        if max(k, n) > np.iinfo(np.int32).max:
            raise ValueError("Feature dimensions exceed the CUDA integer limit.")
        if out is None:
            out = cp.empty((k, n, self.p), dtype=cp.float64)
        if (not isinstance(out, cp.ndarray) or out.shape != (k, n, self.p)
                or out.dtype != cp.float64 or not out.flags.c_contiguous
                or out.device.id != self.device_id):
            raise ValueError("out must be C-contiguous float64 with shape (k,n,p).")
        self.kernel(((out.size + 255) // 256,), (256,), (
            np.int32(k), np.int32(n), np.int32(d), np.int32(self.p), X, self.a, self.b, out,
        ))
        return out


def build_features(X, *, degree=2, include_bias=True, bias_last=False):
    """Allocate a design for one call; use FeaturePlan to reuse basis/output.

    Polynomial order matches sklearn degree 1/2; bias_last=True selects the
    affine model's order. X retains its C-contiguous float64 (k,n,d) layout.
    """
    if X.ndim != 3 or min(X.shape) < 1:
        raise ValueError("X must have nonempty shape (k,n,d).")
    return FeaturePlan(X.shape[2], degree=degree, include_bias=include_bias,
                       bias_last=bias_last).build(X)
