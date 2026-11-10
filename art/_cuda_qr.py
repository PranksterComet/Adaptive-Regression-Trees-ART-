"""Reusable float64 CUDA QR solves; imported only by the optional GPU optimizer.

Householder factors overwrite A, Q.T is applied to b, and a triangular solve
overwrites b[:p] with the coefficients. Q and a Gram matrix are never formed.
This is an unpivoted, full-column-rank solve, with no implicit CPU fallback.
"""

from __future__ import annotations

import numpy as np


class QrPlan:
    """Own one stream's QR matrix, scratch space, and library handles."""

    def __init__(self, cp, m: int, p: int, stream):
        from cupy_backends.cuda.libs import cublas, cusolver

        if not 1 <= p <= m:
            raise ValueError("GPU QR requires at least as many rows as columns.")
        self.cp, self.m, self.p, self.stream = cp, m, p, stream
        self.blas, self.solver = cublas, cusolver
        self.blas_handle = self.solver_handle = None
        with stream:
            self.a = cp.empty((m, p), dtype=cp.float64, order="F")
            self.b = cp.empty(m, dtype=cp.float64)
            self.tau = cp.empty(p, dtype=cp.float64)
            self.info = cp.empty(2, dtype=cp.int32)
            self.one = np.array(1.0, dtype=np.float64)
            self.blas_handle = cublas.create()
            self.solver_handle = cusolver.create()
            cublas.setStream(self.blas_handle, stream.ptr)
            cusolver.setStream(self.solver_handle, stream.ptr)
            qr_size = cusolver.dgeqrf_bufferSize(
                self.solver_handle, m, p, self.a.data.ptr, m
            )
            qt_size = cusolver.dormqr_bufferSize(
                self.solver_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_OP_T,
                m, 1, p, self.a.data.ptr, m, self.tau.data.ptr, self.b.data.ptr, m,
            )
            self.work = cp.empty(max(qr_size, qt_size), dtype=cp.float64)

    @property
    def coefficients(self):
        return self.b[:self.p]

    def solve(self) -> None:
        """Enqueue the solve; info and rank checks stay on the device."""
        s, b = self.solver, self.blas
        m, p, handle = self.m, self.p, self.solver_handle
        with self.stream:
            s.dgeqrf(handle, m, p, self.a.data.ptr, m, self.tau.data.ptr,
                     self.work.data.ptr, self.work.size, self.info.data.ptr)
            s.dormqr(handle, b.CUBLAS_SIDE_LEFT, b.CUBLAS_OP_T, m, 1, p,
                     self.a.data.ptr, m, self.tau.data.ptr, self.b.data.ptr, m,
                     self.work.data.ptr, self.work.size,
                     self.info.data.ptr + self.info.itemsize)
            b.dtrsm(self.blas_handle, b.CUBLAS_SIDE_LEFT, b.CUBLAS_FILL_MODE_UPPER,
                    b.CUBLAS_OP_N, b.CUBLAS_DIAG_NON_UNIT, p, 1,
                    self.one.ctypes.data, self.a.data.ptr, m, self.b.data.ptr, m)

    def close(self) -> None:
        if self.solver_handle is not None:
            self.solver.destroy(self.solver_handle)
            self.solver_handle = None
        if self.blas_handle is not None:
            self.blas.destroy(self.blas_handle)
            self.blas_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # The CUDA runtime may already have shut down.


def cuda_qr_status() -> tuple[bool, str]:
    """Probe QR, a triangular solve, and the custom kernels on real hardware."""
    try:
        import cupy as cp
        from ._cuda_optimizer_kernels import kernels

        stream = cp.cuda.Stream(non_blocking=True)
        plan = QrPlan(cp, 3, 2, stream)
        try:
            with stream:
                plan.a.set(np.array([[1., 0.], [0., 1.], [1., 1.]], order="F"))
                plan.b.set(np.array([1., 2., 3.]))
                plan.solve()
                np.testing.assert_allclose(plan.coefficients.get(stream=stream), [1., 2.])
                # Compile every kernel, then run one gate calculation as a smoke test.
                functions = kernels(cp.cuda.Device().id)
                theta = cp.zeros(2)
                u = cp.zeros(1)
                pi, q, sq = cp.empty(1), cp.empty(2), cp.empty(2)
                functions["gates"]((1,), (256,), (
                    np.int32(1), np.int32(1), np.float64(1), np.float64(0),
                    np.float64(0), np.int32(0), np.int32(0), theta, u, u, u, pi, q, sq,
                ))
                np.testing.assert_allclose(pi.get(stream=stream), [.5])
        finally:
            stream.synchronize()
            plan.close()
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return True, f"{name}; CuPy {cp.__version__}; cuSOLVER QR"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
