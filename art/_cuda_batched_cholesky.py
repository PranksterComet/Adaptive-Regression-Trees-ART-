"""Resident, batched left/right weighted ridge fits for equally sized nodes.

CuPy is imported only when a workspace is constructed. Subset fits compact
pointer tables while cached features and results stay in fixed node slots.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def _dimensions(*shape):
    if any(not isinstance(v, (int, np.integer)) or not 0 < v <= np.iinfo(np.int32).max
           for v in shape):
        raise ValueError("Workspace dimensions must be positive CUDA integers.")
    return tuple(int(v) for v in shape)


def _node_ids(nodes, limit):
    ids = np.asarray(nodes)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError("nodes must be a one-dimensional integer list.")
    if np.any(ids < 0) or np.any(ids >= limit) or len(np.unique(ids)) != len(ids):
        raise ValueError("nodes must contain distinct loaded slot indices.")
    return np.ascontiguousarray(ids, dtype=np.int32)


@lru_cache(maxsize=None)
def _kernels(device_id: int):
    import cupy as cp

    names = ("batch_check_data", "batch_weights", "batch_regularize", "batch_protect", "batch_finish",
             "batch_weights_active", "batch_weighted_inputs", "batch_system_ptrs",
             "batch_regularize_active", "batch_protect_active", "batch_finish_active")
    module = cp.RawModule(
        code=Path(__file__).with_name("_cuda_batched_cholesky_kernels.cu").read_text(),
        options=("--std=c++11",), name_expressions=names,
    )
    return {name: module.get_function(name) for name in names}


class BatchedCholeskyWorkspace:
    """Cache Phi (K,n,p), y (K,n), and buffers for 2K float64 solves.

    ``fit(pi)`` accepts gates of shape (K,n) and returns device views
    ``(coefficients, status)`` of shapes (K,2,p) and (K,2). Side 0 is left.
    These views are overwritten by the next fit; copy them to retain a result.

    Status is 0 on success, positive for a non-positive Cholesky pivot (its
    one-based index), -1 for a nonfinite/out-of-range gate, -2 for nonfinite
    assembly/factors, and -3 for a failed/nonfinite solve. Failed coefficients
    are NaN. There is no condition estimate or numerical fallback.

    Calls enqueue work without downloading statuses or synchronizing. Inputs
    must be ready on ``stream`` (the current stream at construction by default).
    Consume results on that stream, or call ``synchronize()`` first. A workspace
    belongs to one device/stream and must not be used concurrently.

    By default Phi/y are copied once. ``copy=False`` borrows C-contiguous
    float64 device arrays without allocating duplicates. Keep their contents
    fixed during fits. allocate(K,n,p) creates empty reusable capacity;
    load(Phi,y) fills k<=K slots without changing any addresses or strides.
    """

    def __init__(self, phi, y, ridge: float = 0.0, weight_floor: float = 1e-12, *,
                 stream=None, copy=True):
        shape = np.shape(phi)
        if len(shape) != 3 or any(size < 1 for size in shape):
            raise ValueError("Phi must have nonempty shape (K, n, p).")
        if np.shape(y) != shape[:2]:
            raise ValueError("y must have shape (K, n).")
        if np.iscomplexobj(phi) or np.iscomplexobj(y):
            raise ValueError("Phi and y must be real.")
        if not isinstance(copy, (bool, np.bool_)):
            raise ValueError("copy must be a boolean.")
        self._allocate(shape, ridge, weight_floor, stream, phi, y, copy)

    @classmethod
    def allocate(cls, capacity, n, p, ridge=0., weight_floor=1e-12, *, stream=None):
        """Reserve all numerical buffers and pointer plans before loading data."""
        ws = cls.__new__(cls)
        ws._allocate(_dimensions(capacity, n, p), ridge, weight_floor, stream)
        with ws.stream:
            ws._prepare_products()
        return ws

    def _allocate(self, shape, ridge, weight_floor, stream, phi=None, y=None, copy=True):
        self.K, self.n, self.p = _dimensions(*shape)
        self.capacity, self.k = self.K, 0 if phi is None else self.K
        self.ridge, self.weight_floor = float(ridge), float(weight_floor)
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative.")
        if not np.isfinite(self.weight_floor) or self.weight_floor < 0:
            raise ValueError("weight_floor must be finite and nonnegative.")
        import cupy as cp
        from cupy_backends.cuda.libs import cublas, cusolver

        self.cp, self.solver, self.handle = cp, cusolver, None
        self.blas, self.blas_handle = cublas, None
        self._nodes, self._products = None, None
        self.last_batch_size = 0
        self.fit_count = 0
        self.device_id = cp.cuda.Device().id
        self.stream = cp.cuda.get_current_stream() if stream is None else stream
        self.lower = cublas.CUBLAS_FILL_MODE_LOWER
        with self.stream:
            if phi is None:
                self.phi = cp.empty(shape, dtype=cp.float64)
                self.y = cp.empty(shape[:2], dtype=cp.float64)
            elif copy:
                # Own the data: caller mutations cannot invalidate cached fits.
                self.phi = cp.array(phi, dtype=cp.float64, order="C", copy=True)
                self.y = cp.array(y, dtype=cp.float64, order="C", copy=True)
            else:
                if any(not isinstance(a, cp.ndarray) or a.dtype != cp.float64
                       or not a.flags.c_contiguous or a.device.id != self.device_id
                       for a in (phi, y)):
                    raise ValueError("copy=False requires C-contiguous float64 device Phi/y.")
                self.phi, self.y = phi, y
            self.functions = _kernels(self.device_id)
            self._phi_t = self.phi.swapaxes(-1, -2)
            # Side-major storage keeps each side's batched matrix products contiguous.
            self.weights = cp.empty((2, self.K, self.n), dtype=cp.float64)
            self._weighted_phi = cp.empty_like(self.phi)
            self._weighted_y = cp.empty_like(self.y)
            self.grams = cp.empty((2, self.K, self.p, self.p), dtype=cp.float64)
            self._factor_storage = cp.empty_like(self.grams)
            # Each individual matrix, rather than the entire stack, is column-major.
            # Only its lower triangle is meaningful after factorization/solve.
            self.factors = self._factor_storage.swapaxes(-1, -2)
            self._rhs = cp.empty((2, self.K, self.p), dtype=cp.float64)
            self._ridge = cp.empty((2, self.K), dtype=cp.float64)
            self._status = cp.empty((2, self.K), dtype=cp.int32)
            self._invalid = cp.empty(self.K, dtype=cp.int32)
            self._invalid_host = np.empty(self.K, dtype=np.int32)
            self._factor_info = cp.empty(2 * self.K, dtype=cp.int32)
            self._solve_info = cp.empty(1, dtype=cp.int32)  # potrsBatched reports one scalar.
            self._pi = cp.empty((self.K, self.n), dtype=cp.float64)
            self._all_ids = np.arange(self.K, dtype=np.int32)
            self.coefficients = self._rhs.transpose(1, 0, 2)
            self.status = self._status.T
            self.ridge_effective = self._ridge.T
            # Native batched cuSOLVER takes device arrays of matrix/vector pointers.
            offsets = np.arange(2 * self.K, dtype=np.uintp)
            self._a_ptrs = cp.asarray(self._factor_storage.data.ptr + offsets * self.p**2 * 8)
            self._b_ptrs = cp.asarray(self._rhs.data.ptr + offsets * self.p * 8)
            if phi is not None:
                self._check_data(self.K)
            self.handle = cusolver.create()
            cusolver.setStream(self.handle, self.stream.ptr)

    def load(self, phi, y):
        """Replace k equally sized datasets in place, preserving all plans.

        Inputs can be host/device arrays. A view of self.phi[:k], already filled
        by a feature kernel, is accepted without copying. No fits may overlap.
        """
        self._check_open()
        shape = np.shape(phi)
        if (len(shape) != 3 or not 0 < shape[0] <= self.K
                or shape[1:] != (self.n, self.p) or np.shape(y) != shape[:2]):
            raise ValueError("Phi/y must have shapes (k,n,p)/(k,n), 0<k<=capacity.")
        if np.iscomplexobj(phi) or np.iscomplexobj(y):
            raise ValueError("Phi and y must be real.")
        self.k = 0  # A failed upload/validation must not leave a usable stale batch.
        cp, k = self.cp, shape[0]
        with self.stream:
            for dst, src in ((self.phi[:k], phi), (self.y[:k], y)):
                if isinstance(src, cp.ndarray):
                    if src.device.id != self.device_id:
                        raise ValueError("Inputs must be on the workspace's device.")
                    if src.data.ptr != dst.data.ptr or src.strides != dst.strides or src.dtype != dst.dtype:
                        cp.copyto(dst, src)
                else:
                    dst.set(np.ascontiguousarray(src, dtype=np.float64), stream=self.stream)
            self._check_data(k)
        self.k, self.last_batch_size, self.fit_count = k, 0, 0
        return self

    def _check_data(self, k):
        self._invalid.fill(0)
        self.functions["batch_check_data"](((k * self.n * self.p + 255) // 256,), (256,), (
            np.int32(k), np.int32(self.n), np.int32(self.p), self.phi, self.y, self._invalid,
        ))
        self._invalid.get(stream=self.stream, out=self._invalid_host)
        if self._invalid_host[:k].any():
            raise ValueError("Phi and y must contain only finite values.")

    def _check_open(self):
        if self.handle is None:
            raise RuntimeError("The workspace is closed.")
        if self.cp.cuda.Device().id != self.device_id:
            raise ValueError("Use the workspace on the device where it was created.")

    def _assemble(self, pi, floor=None) -> None:
        """Form both weighted Grams and RHS vectors directly, batched over nodes."""
        cp = self.cp
        self._invalid.fill(0)
        self.functions["batch_weights"](((self.K * self.n + 255) // 256,), (256,), (
            np.int32(self.K), np.int32(self.n), np.float64(self.weight_floor if floor is None else floor),
            pi, self.weights, self._invalid,
        ))
        for side in range(2):
            cp.multiply(self.phi, self.weights[side, :, :, None], out=self._weighted_phi)
            cp.matmul(self._phi_t, self._weighted_phi, out=self.grams[side])
            cp.multiply(self.y, self.weights[side], out=self._weighted_y)
            cp.matmul(self._phi_t, self._weighted_y[:, :, None], out=self._rhs[side, :, :, None])
        self.functions["batch_regularize"]((2 * self.K,), (256,), (
            np.int32(self.K), np.int32(self.p), np.float64(self.ridge),
            self.grams, self._invalid, self._factor_storage, self._rhs,
            self._ridge, self._status,
        ))

    def _prepare_products(self):
        """Allocate indexed GEMM plans once, eagerly for capacity workspaces."""
        cp = self.cp
        if self._products is None:
            from ._cuda_batched_matmul import BatchedMatmul

            self.ids = cp.empty(self.K, dtype=cp.int32)
            self._ap = cp.empty_like(self._a_ptrs)
            self._bp = cp.empty_like(self._b_ptrs)
            self.blas_handle = self.blas.create()
            self.blas.setStream(self.blas_handle, self.stream.ptr)
            self._products = []
            for side in range(2):
                self._products.append((
                    BatchedMatmul(cp, self._phi_t, self._weighted_phi, self.grams[side],
                                  self.blas_handle, self.stream),
                    BatchedMatmul(cp, self._phi_t, self._weighted_y[:, :, None],
                                  self._rhs[side, :, :, None], self.blas_handle, self.stream),
                ))

    def _select(self, nodes):
        """Prepare active prefixes only when the CPU node list changes."""
        self._prepare_products()
        if self._nodes is not None and np.array_equal(nodes, self._nodes):
            return
        self._nodes = nodes.copy()
        count = len(nodes)
        self.ids[:count].set(nodes, stream=self.stream)
        for pair in self._products:
            for plan in pair:
                plan.select(self.ids, count)
        self.functions["batch_system_ptrs"](((2 * count + 255) // 256,), (256,), (
            np.int32(self.K), np.int32(count), self.ids, self._a_ptrs, self._b_ptrs,
            self._ap, self._bp,
        ))

    def _assemble_active(self, pi, count, floor):
        self._invalid.fill(0)
        self.functions["batch_weights_active"](((count * self.n + 255) // 256,), (256,), (
            np.int32(self.K), np.int32(self.n), np.int32(count), np.float64(floor),
            self.ids, pi, self.weights, self._invalid,
        ))
        for side, (gram, rhs) in enumerate(self._products):
            self.functions["batch_weighted_inputs"](((count * self.n * self.p + 255) // 256,), (256,), (
                np.int32(self.K), np.int32(self.n), np.int32(self.p), np.int32(count),
                np.int32(side), self.ids, self.phi, self.y, self.weights,
                self._weighted_phi, self._weighted_y,
            ))
            gram.run()
            rhs.run()
        self.functions["batch_regularize_active"]((2 * count,), (256,), (
            np.int32(self.K), np.int32(self.p), np.int32(count), np.float64(self.ridge),
            self.ids, self.grams, self._invalid, self._factor_storage, self._rhs,
            self._ridge, self._status,
        ))

    def fit(self, pi, *, nodes=None, weight_floor=None):
        """Enqueue direct Gram assembly, native batched Cholesky, and solves.

        An already contiguous float64 CuPy gate array needs no conversion or
        transfer. NumPy gates are also accepted and uploaded for this call.
        nodes is an optional CPU list of distinct slot indices. Other slots
        retain coefficients/status/ridge. weight_floor=0 enables hard refits.
        """
        self._check_open()
        if not self.k:
            raise RuntimeError("Load data before fitting.")
        cp, solver, p = self.cp, self.solver, self.p
        if np.shape(pi) not in ((self.K, self.n), (self.k, self.n)) or np.iscomplexobj(pi):
            raise ValueError("pi must be real with shape (k,n) or (capacity,n).")
        if isinstance(pi, cp.ndarray) and pi.device.id != self.device_id:
            raise ValueError("pi must be on the workspace's device.")
        floor = self.weight_floor if weight_floor is None else float(weight_floor)
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("weight_floor must be finite and nonnegative.")
        if nodes is not None:
            nodes = _node_ids(nodes, self.k)
        elif self.k < self.K:
            nodes = self._all_ids[:self.k]
        count = self.k if nodes is None else len(nodes)
        self.last_batch_size = 2 * count
        if not count:
            return self.coefficients, self.status
        self.fit_count += count
        with self.stream:
            if isinstance(pi, cp.ndarray):
                pi = cp.asarray(pi, dtype=cp.float64, order="C")
            else:
                rows = np.shape(pi)[0]
                self._pi[:rows].set(np.ascontiguousarray(pi, dtype=np.float64), stream=self.stream)
                pi = self._pi
            if nodes is None:
                self._assemble(pi, floor)
                ap, bp = self._a_ptrs, self._b_ptrs
            else:
                self._select(nodes)
                self._assemble_active(pi, count, floor)
                ap, bp = self._ap, self._bp
            solver.dpotrfBatched(self.handle, self.lower, p, ap.data.ptr,
                                 p, self._factor_info.data.ptr, 2 * count)
            # Replace failed factors/RHS with I/0 before the common solve. The
            # failure status survives; the corresponding final coefficients are NaN.
            prefix = () if nodes is None else (np.int32(self.K),)
            suffix = () if nodes is None else (np.int32(count), self.ids)
            tag = "" if nodes is None else "_active"
            self.functions["batch_protect" + tag]((2 * count,), (256,), (*prefix,
                np.int32(p), *suffix, self._factor_storage, self._rhs,
                self._factor_info, self._status,
            ))
            solver.dpotrsBatched(self.handle, self.lower, p, 1, ap.data.ptr,
                                 p, bp.data.ptr, p, self._solve_info.data.ptr,
                                 2 * count)
            self.functions["batch_finish" + tag]((2 * count,), (256,), (*prefix,
                np.int32(p), *suffix, self._rhs, self._solve_info, self._status,
            ))
        return self.coefficients, self.status

    def synchronize(self) -> None:
        """Wait until this workspace's queued operations have completed."""
        self.stream.synchronize()

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            with self.cp.cuda.Device(self.device_id):
                self.synchronize()
                self.solver.destroy(self.handle)
                self.handle = None
                if self.blas_handle is not None:
                    self.blas.destroy(self.blas_handle)
                    self.blas_handle = None
                self._products = None
                for name, value in list(vars(self).items()):
                    if isinstance(value, self.cp.ndarray):
                        setattr(self, name, None)
                self.k = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # The CUDA runtime may already have shut down.
