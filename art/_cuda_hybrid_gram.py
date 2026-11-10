"""GPU Gram/RHS construction for the hybrid GPU/CPU optimizer.

CuPy is imported lazily. All public inputs and outputs are NumPy arrays; CPU
model fitting and tree serialization never retain a device array. A workspace
belongs to one ordered dataset and is not intended for concurrent fits.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import WeightedNormalEquations


def cuda_status() -> tuple[bool, str]:
    """Probe an actual float64 cuBLAS operation, not just the CUDA driver."""
    try:
        import cupy as cp

        a = cp.ones((2, 2), dtype=cp.float64)
        np.testing.assert_allclose(cp.asnumpy(a @ a), 2.0)
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return True, f"{name}; CuPy {cp.__version__}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class CudaWeightedGram:
    """Cache Phi/y once and construct left/right systems from raw CPU gates.

    Initialization is deferred until the first fit, so uploads, precomputation,
    and allocations are included in the splitter's least-squares timing.
    """

    def __init__(self, design: np.ndarray, y: np.ndarray, weight_floor: float):
        self.design = np.asarray(design, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if self.design.ndim != 2 or self.design.shape[0] != self.y.size:
            raise ValueError("CUDA design and target dimensions do not match.")
        if not np.isfinite(weight_floor) or weight_floor < 0.0:
            raise ValueError("CUDA weight_floor must be finite and nonnegative.")
        self.weight_floor = float(weight_floor)
        self._cp = None
        self.setup_seconds = 0.0
        self.direct_rebuilds = 0

    def validate(self, design: np.ndarray, y: np.ndarray, weight_floor: float) -> None:
        """Reject stale workspaces, including different data with equal shapes."""
        if (
            self.weight_floor != weight_floor
            or self.design.shape != design.shape
            or not (self.design is design or np.array_equal(self.design, design))
            or not np.array_equal(self.y, np.asarray(y).reshape(-1))
        ):
            raise ValueError("CUDA workspace belongs to a different design, target, or floor.")

    def _initialize(self) -> None:
        if self._cp is not None:
            return
        start = perf_counter()
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "optimizer='hybrid_gram' requires a working CuPy/CUDA installation."
            ) from exc
        # Store Phi in column-major order for the repeated BLAS products.
        self._phi = cp.asarray(self.design, dtype=cp.float64, order="F")
        self._y = cp.asarray(self.y, dtype=cp.float64)
        n, p = self.design.shape
        self._g0 = self._phi.T @ self._phi
        self._h0 = self._phi.T @ self._y
        self._g0_norm = float(cp.linalg.norm(self._g0, ord=1).get())
        self._h0_norm = float(cp.linalg.norm(self._h0, ord=1).get())
        self._row_energy = np.einsum("ij,ij->i", self.design, self.design)
        self._pi = cp.empty(n, dtype=cp.float64)
        self._weights = cp.empty((2, n), dtype=cp.float64)
        self._weighted_phi = cp.empty_like(self._phi)
        self._weighted_y = cp.empty(n, dtype=cp.float64)
        self._grams = cp.empty((2, p, p), dtype=cp.float64)
        self._rhs = cp.empty((2, p), dtype=cp.float64)
        self._stream = cp.cuda.get_current_stream()
        self._stream.synchronize()
        self._cp = cp
        self.setup_seconds = perf_counter() - start

    def _set_weights(self, pi: np.ndarray) -> None:
        cp = self._cp
        self._pi.set(pi)
        cp.subtract(1.0, self._pi, out=self._weights[0])
        cp.maximum(self._weights[0], self.weight_floor, out=self._weights[0])
        cp.maximum(self._pi, self.weight_floor, out=self._weights[1])

    def _direct(self, side: int) -> None:
        cp = self._cp
        q = self._weights[side]
        cp.multiply(self._phi, q[:, None], out=self._weighted_phi)
        cp.matmul(self._phi.T, self._weighted_phi, out=self._grams[side])
        cp.multiply(self._y, q, out=self._weighted_y)
        cp.matmul(self._phi.T, self._weighted_y, out=self._rhs[side])

    def _rebuild(self, pi: np.ndarray, side: int) -> WeightedNormalEquations:
        self.direct_rebuilds += 1
        self._set_weights(pi)
        self._direct(side)
        return WeightedNormalEquations(
            self._cp.asnumpy(self._grams[side]),
            self._cp.asnumpy(self._rhs[side]),
        )

    def form_pair(
        self, pi: np.ndarray
    ) -> tuple[WeightedNormalEquations, WeightedNormalEquations]:
        pi = np.ascontiguousarray(pi, dtype=float)
        if pi.shape != self.y.shape or not np.all(np.isfinite(pi)):
            raise ValueError("CUDA sigmoid vector must be finite and match y.")
        if np.any(pi < 0.0) or np.any(pi > 1.0):
            raise ValueError("CUDA sigmoid values must be in [0, 1].")
        self._initialize()
        cp = self._cp
        self._set_weights(pi)

        # Floor corrections are nonzero only on clipped rows. If many rows are
        # clipped, two direct products cost less than forming the correction.
        clipped = np.flatnonzero((pi < self.weight_floor) | (1.0 - pi < self.weight_floor))
        derived = None
        if clipped.size > pi.size // 4:
            self._direct(0)
            self._direct(1)
        else:
            # Compute the smaller side directly, avoiding the worst cancellation
            # when one side has nearly all of the weighted feature energy.
            direct = 1 if pi @ self._row_energy <= 0.5 * self._row_energy.sum() else 0
            derived = 1 - direct
            self._direct(direct)
            cp.subtract(self._g0, self._grams[direct], out=self._grams[derived])
            cp.subtract(self._h0, self._rhs[direct], out=self._rhs[derived])
            if clipped.size:
                # qL + qR = 1 + delta, exactly accounting for both floors.
                rows = cp.asarray(clipped)
                block = self._phi[rows]
                raw = self._pi[rows]
                delta = cp.maximum(self.weight_floor - raw, 0.0) + cp.maximum(
                    self.weight_floor - (1.0 - raw), 0.0
                )
                self._grams[derived] += block.T @ (block * delta[:, None])
                self._rhs[derived] += block.T @ (self._y[rows] * delta)

        # Blocking downloads ensure all GPU work is included in caller timings.
        grams = cp.asnumpy(self._grams)
        rhs = cp.asnumpy(self._rhs)
        systems = [WeightedNormalEquations(grams[i], rhs[i]) for i in (0, 1)]
        if derived is not None:
            # Guard subtraction error in the CPU condition check. The bound is
            # for subtraction roundoff, not a full GEMM forward-error estimate.
            saved_pi = pi.copy()
            systems[derived] = WeightedNormalEquations(
                grams[derived], rhs[derived],
                subtraction_bound=8 * np.finfo(float).eps * self._g0_norm,
                rebuild=lambda: self._rebuild(saved_pi, derived),
                subtraction_rhs_bound=8 * np.finfo(float).eps * (
                    self._h0_norm + np.linalg.norm(rhs[direct], 1)
                ),
            )
        return systems[0], systems[1]
