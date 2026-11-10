"""Reusable pointer-based float64 matrix products for fixed GPU node slots."""

from functools import lru_cache
from pathlib import Path

import numpy as np


def _transpose_layout(shape, strides):
    """Describe A.T to column-major BLAS without moving A's entries."""
    r, c = shape
    sr, sc = strides
    if (c == 1 or sc == 8) and (r == 1 or sr >= c * 8):
        return False, max(c, sr // 8)
    if (r == 1 or sr == 8) and (c == 1 or sc >= r * 8):
        return True, max(r, sc // 8)
    raise ValueError("Each matrix must have a positive row- or column-major layout.")


@lru_cache(maxsize=None)
def _select_kernel(device_id):
    import cupy as cp

    return cp.RawModule(
        code=Path(__file__).with_suffix(".cu").read_text(),
        options=("--std=c++11",),
    ).get_function("select_matmul_ptrs")


class BatchedMatmul:
    """Compute out=A@B for all slots, or a selected pointer prefix.

    Inputs are float64 device views (K,m,k)/(K,k,n); out is row-major
    (K,m,n). Padded strides and vector views are supported. The caller owns
    the BLAS handle/stream and keeps all arrays alive. select() follows stream
    order; only pointer tables move, never matrices.
    """

    def __init__(self, cp, a, b, out, handle, stream):
        from cupy_backends.cuda.libs import cublas

        if any(v.ndim != 3 or v.dtype != cp.float64 for v in (a, b, out)):
            raise ValueError("Batched products require three-dimensional float64 arrays.")
        K, m, k = a.shape
        if b.shape[:2] != (K, k) or out.shape != (K, m, b.shape[2]):
            raise ValueError("Batched product dimensions do not match.")
        ta, self.lda = _transpose_layout(a.shape[1:], a.strides[1:])
        tb, self.ldb = _transpose_layout(b.shape[1:], b.strides[1:])
        tc, self.ldc = _transpose_layout(out.shape[1:], out.strides[1:])
        if tc:
            raise ValueError("Output matrices must be row-major.")
        self.cp, self.blas, self.handle, self.stream = cp, cublas, handle, stream
        self.a, self.b, self.out = a, b, out
        self.K, self.m, self.n, self.k = K, m, b.shape[2], k
        self.ta = cublas.CUBLAS_OP_T if ta else cublas.CUBLAS_OP_N
        self.tb = cublas.CUBLAS_OP_T if tb else cublas.CUBLAS_OP_N
        offsets = np.arange(K, dtype=np.uintp)
        with stream:
            self.base = cp.asarray(np.stack([v.data.ptr + offsets * v.strides[0]
                                             for v in (a, b, out)]))
            self.ptrs = cp.empty_like(self.base)
            self.select_kernel = _select_kernel(cp.cuda.Device().id)
        self.one, self.zero = np.array(1., dtype=np.float64), np.array(0., dtype=np.float64)
        self.table, self.count = self.base, K

    def select(self, ids=None, count=None):
        if ids is None:
            self.table, self.count = self.base, self.K
            return
        self.count = int(count)
        self.table = self.ptrs
        if self.count:
            with self.stream:
                self.select_kernel(((self.count + 255) // 256,), (256,), (
                    np.int32(self.K), np.int32(self.count), ids, self.base, self.ptrs,
                ))

    def run(self):
        if not self.count:
            return
        # Row-major C=A@B is column-major C.T=B.T@A.T.
        ptr = self.table.data.ptr
        with self.stream:
            self.blas.dgemmBatched(
                self.handle, self.tb, self.ta, self.n, self.m, self.k,
                self.one.ctypes.data, ptr + self.K * 8, self.ldb,
                ptr, self.lda, self.zero.ctypes.data,
                ptr + 2 * self.K * 8, self.ldc, self.count,
            )
