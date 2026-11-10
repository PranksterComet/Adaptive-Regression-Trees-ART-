"""Reusable CUDA normal-equation fits with complementary weighted Grams.

No condition estimator or numerical fallback is provided. Device-side status
checks belong to the surrounding optimizer, just as for the QR plans.
"""

from __future__ import annotations

import numpy as np


class CholeskyPlan:
    """Own one stream's regularized Gram, RHS, and cuSOLVER workspace."""

    def __init__(self, cp, p: int, stream):
        from cupy_backends.cuda.libs import cublas, cusolver

        self.p, self.stream = p, stream
        self.solver = cusolver
        self.handle = None
        self.lower = cublas.CUBLAS_FILL_MODE_LOWER
        with stream:
            self.a = cp.empty((p, p), dtype=cp.float64, order="F")
            self.b = cp.empty(p, dtype=cp.float64)
            self.info = cp.empty(2, dtype=cp.int32)
            self.handle = cusolver.create()
            cusolver.setStream(self.handle, stream.ptr)
            size = cusolver.dpotrf_bufferSize(self.handle, self.lower, p, self.a.data.ptr, p)
            self.work = cp.empty(size, dtype=cp.float64)

    @property
    def coefficients(self):
        return self.b

    def solve(self) -> None:
        """Enqueue factorization and triangular solves without a host status copy."""
        s, p = self.solver, self.p
        with self.stream:
            s.dpotrf(self.handle, self.lower, p, self.a.data.ptr, p,
                     self.work.data.ptr, self.work.size, self.info.data.ptr)
            s.dpotrs(self.handle, self.lower, p, 1, self.a.data.ptr, p,
                     self.b.data.ptr, p, self.info.data.ptr + self.info.itemsize)

    def close(self) -> None:
        if self.handle is not None:
            self.solver.destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # The CUDA runtime may already have shut down.


class ComplementaryGrams:
    """Assemble two unregularized systems before either factorization starts.

    Fixed Phi/y and mutable buffers remain on the device. A small trace packet
    selects the directly formed side. Both RHS vectors are computed directly.
    """

    def __init__(self, cp, phi, target):
        self.cp, self.phi, self.target = cp, phi, target
        self.total = phi.T @ phi
        self.weighted_phi = cp.empty_like(phi)
        self.weighted_y = cp.empty_like(target)

    def assemble(self, plans, q, pi, traces, floor, stream):
        cp, phi = self.cp, self.phi
        left_trace, right_trace = traces.get(stream=stream)
        direct = 1 if right_trace <= left_trace else 0
        derived = 1 - direct
        cp.multiply(phi, q[direct, :, None], out=self.weighted_phi)
        cp.matmul(phi.T, self.weighted_phi, out=plans[direct].a)
        cp.subtract(self.total, plans[direct].a, out=plans[derived].a)

        if floor > 0:
            rows = cp.flatnonzero((pi < floor) | (1.0 - pi < floor))
            if rows.size:
                # qL + qR = 1 + delta, including when both weights are floored.
                raw = pi[rows]
                delta = cp.maximum(floor - raw, 0.0) + cp.maximum(floor - (1.0 - raw), 0.0)
                block = phi[rows]
                plans[derived].a += block.T @ (block * delta[:, None])

        for side, plan in enumerate(plans):
            cp.multiply(self.target, q[side], out=self.weighted_y)
            cp.matmul(phi.T, self.weighted_y, out=plan.b)


def cuda_cholesky_status() -> tuple[bool, str]:
    """Probe cuSOLVER Cholesky and the shared custom-kernel compilation."""
    try:
        import cupy as cp
        from ._cuda_optimizer_kernels import kernels

        stream = cp.cuda.Stream(non_blocking=True)
        plan = CholeskyPlan(cp, 2, stream)
        try:
            with stream:
                plan.a.set(np.array([[4., 1.], [1., 3.]], order="F"))
                plan.b.set(np.array([6., 7.]))
                plan.solve()
                np.testing.assert_allclose(plan.coefficients.get(stream=stream), [1., 2.])
                np.testing.assert_array_equal(plan.info.get(stream=stream), [0, 0])
                functions = kernels(cp.cuda.Device().id)
                status = cp.empty(2, dtype=cp.int32)
                functions["cholesky_check"]((1,), (256,), (
                    np.int32(2), np.int32(0), plan.a, plan.b, plan.info, status,
                ))
                assert status[:1].get(stream=stream)[0] == 0
        finally:
            stream.synchronize()
            plan.close()
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return True, f"{name}; CuPy {cp.__version__}; cuSOLVER Cholesky"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
