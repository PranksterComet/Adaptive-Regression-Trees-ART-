"""Host execution of production CUDA kernels with NumPy BLAS/solver/RNG stand-ins.

This exercises orchestration, layouts and pointer selection, not CUDA runtime
behavior or performance. Real-device tests run the same workflows separately.
"""

import ctypes
from itertools import count
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_cuda_optimizer_kernels import compile_kernel_source


class Array(np.ndarray):
    reads = []

    @property
    def data(self):
        return SimpleNamespace(ptr=self.ctypes.data)

    @property
    def device(self):
        return SimpleNamespace(id=0)

    def get(self, stream=None, out=None):
        self.reads.append(tuple(self.shape))
        if out is not None:
            np.copyto(out, self)
            return out
        return np.array(self)

    def set(self, value, stream=None):
        np.copyto(self, value)


class Stream:
    ptr = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def synchronize(self):
        pass


class Device(Stream):
    def __init__(self, device_id=0):
        self.id = device_id


def vector(ptr, count, dtype=ctypes.c_double):
    return np.ctypeslib.as_array((dtype * count).from_address(int(ptr)))


def matrix(ptr, rows, cols, ld):
    data = vector(ptr, ld * (cols - 1) + rows)
    return np.ndarray((rows, cols), dtype=np.float64, buffer=data, strides=(8, ld * 8))


def install(monkeypatch, tmp_path_factory):
    sources = ("_cuda_optimizer_kernels.cu", "_cuda_batched_cholesky_kernels.cu",
               "_cuda_batched_matmul.cu", "_cuda_batched_optimizer.cu", "_cuda_features.cu")
    from pathlib import Path

    root = Path(__file__).parents[1] / "art"
    compiled = {(root / name).read_text(): compile_kernel_source(tmp_path_factory, name)
                for name in sources}

    class RawModule:
        def __init__(self, code, **kwargs):
            self.run = compiled[code]

        def get_function(self, name):
            return lambda grid, block, args: self.run(name, grid, *args)

    calls = []

    def gemm(handle, ta, tb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc, count):
        av, bv, cv = (vector(ptr, count, ctypes.c_uint64) for ptr in (ap, bp, cp))
        a, b = vector(alpha, 1)[0], vector(beta, 1)[0]
        calls.append(("gemm", count, m, n, k))
        for i in range(count):
            A = matrix(av[i], k if ta else m, m if ta else k, lda)
            B = matrix(bv[i], n if tb else k, k if tb else n, ldb)
            C = matrix(cv[i], m, n, ldc)
            product = a * ((A.T if ta else A) @ (B.T if tb else B))
            C[...] = product if b == 0 else product + b * C

    def potrf(handle, lower, p, ap, lda, info, count):
        ptrs, flags = vector(ap, count, ctypes.c_uint64), vector(info, count, ctypes.c_int32)
        calls.append(("potrf", count))
        for i, ptr in enumerate(ptrs):
            A = matrix(ptr, p, p, lda)
            symmetric = np.tril(A) + np.tril(A, -1).T
            try:
                L = np.linalg.cholesky(symmetric)
                A[np.tril_indices(p)] = L[np.tril_indices(p)]
                flags[i] = 0
            except np.linalg.LinAlgError:
                for q in range(1, p + 1):
                    try:
                        np.linalg.cholesky(symmetric[:q, :q])
                    except np.linalg.LinAlgError:
                        flags[i] = q
                        break

    def potrs(handle, lower, p, nrhs, ap, lda, bp, ldb, info, count):
        a, b = vector(ap, count, ctypes.c_uint64), vector(bp, count, ctypes.c_uint64)
        calls.append(("potrs", count))
        for i in range(count):
            L, rhs = np.tril(matrix(a[i], p, p, lda)), vector(b[i], p)
            rhs[...] = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        vector(info, 1, ctypes.c_int32)[0] = 0

    blas = SimpleNamespace(create=lambda: 1, destroy=lambda handle: None,
                           setStream=lambda handle, stream: None,
                           CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_FILL_MODE_LOWER=0,
                           dgemmBatched=gemm)
    solver = SimpleNamespace(create=lambda: 1, destroy=lambda handle: None,
                             setStream=lambda handle, stream: None,
                             dpotrfBatched=potrf, dpotrsBatched=potrs)
    rngs, handles = {}, count(1)

    def create_rng(kind):
        handle = next(handles)
        rngs[handle] = None
        return handle

    def seed_rng(handle, seed):
        rngs[handle] = np.random.default_rng(seed)

    def normal(handle, ptr, size, mean, std):
        assert size % 2 == 0
        vector(ptr, size)[...] = rngs[handle].normal(mean, std, size)
        calls.append(("normal", size))

    curand = SimpleNamespace(createGenerator=create_rng, destroyGenerator=rngs.pop,
                             setPseudoRandomGeneratorSeed=seed_rng,
                             setStream=lambda handle, stream: None,
                             generateNormalDouble=normal,
                             CURAND_RNG_PSEUDO_PHILOX4_32_10=161)
    cp = ModuleType("cupy")
    cp.ndarray, cp.float64, cp.int32 = Array, np.float64, np.int32
    cp.asarray = lambda a, **kw: np.asarray(a, **kw).view(Array)
    cp.array = lambda a, **kw: np.array(a, **kw).view(Array)
    for name in ("empty", "empty_like", "zeros", "ones"):
        setattr(cp, name, lambda *a, name=name, **kw: getattr(np, name)(*a, **kw).view(Array))
    cp.copyto, cp.multiply, cp.matmul, cp.sum, cp.isfinite = np.copyto, np.multiply, np.matmul, np.sum, np.isfinite
    cp.median = np.median
    cp.RawModule = RawModule
    cp.cuda = SimpleNamespace(Device=Device, get_current_stream=Stream)
    libs = ModuleType("cupy_backends.cuda.libs")
    libs.cublas, libs.cusolver, libs.curand = blas, solver, curand
    for name in ("cupy_backends", "cupy_backends.cuda"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda.libs", libs)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    cp.calls = calls
    cp.host_reads = Array.reads
    return cp


def clear_caches():
    from art._cuda_batched_cholesky import _kernels
    from art._cuda_batched_matmul import _select_kernel
    from art._cuda_batched_optimizer import _bookkeeping
    from art._cuda_optimizer_kernels import kernels
    from art._cuda_features import _kernel

    for fn in (_kernels, _select_kernel, _bookkeeping, kernels, _kernel):
        fn.cache_clear()


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def cuda_backend(request, tmp_path_factory):
    clear_caches()
    if request.param == "cpu":
        patch = pytest.MonkeyPatch()
        cp = install(patch, tmp_path_factory)
        try:
            yield cp
        finally:
            clear_caches()
            patch.undo()
    else:
        cp = pytest.importorskip("cupy")
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA device unavailable: {exc}")
        if not count:
            pytest.skip("CUDA device unavailable")
        yield cp
        clear_caches()
