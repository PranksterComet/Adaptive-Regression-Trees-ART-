"""Check shared CUDA optimizer kernels against independent NumPy formulas.

On machines without CUDA, compile the same source as C++ and emulate blocks
with host threads and barriers. This checks indexing/reductions and formulas;
it does not validate CUDA libraries, GPU scheduling, or performance.
"""

import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest


def compile_kernel_source(tmp_path_factory, filename):
    """Compile an ART CUDA source with a small host-only launch shim."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for CPU checks of CUDA arithmetic")
    source = (Path(__file__).parents[1] / "art" / filename).read_text()
    signatures = re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', source, re.S)
    shim = r'''
        #include <condition_variable>
        #include <mutex>
        #include <atomic>
        #include <thread>
        #include <vector>
        struct Dim { int x = 0, y = 0, z = 0; };
        thread_local Dim threadIdx, blockIdx;
        Dim blockDim{256, 1}, gridDim;
        // libc++'s polling barrier can back off heavily with 256 host threads.
        // Wake waiters directly while preserving CUDA block-barrier semantics.
        class BlockBarrier {
            std::mutex mutex;
            std::condition_variable ready;
            int arrived = 0;
            unsigned generation = 0;
        public:
            void arrive_and_wait() {
                std::unique_lock lock(mutex);
                unsigned phase = generation;
                if (++arrived == 256) {
                    arrived = 0;
                    ++generation;
                    lock.unlock();
                    ready.notify_all();
                } else ready.wait(lock, [&] { return generation != phase; });
            }
        };
        BlockBarrier* sync_point;
        #define __device__
        #define __forceinline__ inline
        #define __global__
        #define __shared__ static
        #define __syncthreads() sync_point->arrive_and_wait()
        int atomicExch(int* p, int v) { return std::atomic_ref<int>(*p).exchange(v); }
        template<class F> void launch(int gx, int gy, int gz, F f) {
            gridDim = {gx, gy, gz};
            BlockBarrier barrier;
            sync_point = &barrier;
            std::vector<std::thread> workers;
            for (int t = 0; t < 256; ++t) workers.emplace_back([&, t] {
                threadIdx = {t, 0};
                for (int z = 0; z < gz; ++z)
                for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) {
                    blockIdx = {x, y, z};
                    f();
                    barrier.arrive_and_wait();
                }
            });
            for (auto& worker : workers) worker.join();
        }
    '''
    wrappers = []
    declarations = {}
    for name, params in signatures:
        parts = [p.strip() for p in params.split(",")]
        names = [re.search(r"(\w+)$", p).group(1) for p in parts]
        wrappers.append(f'extern "C" void run_{name}(int gx, int gy, int gz, {params}) '
                        f'{{ launch(gx, gy, gz, [&] {{ {name}({", ".join(names)}); }}); }}')
        declarations[name] = parts
    folder = tmp_path_factory.mktemp("cuda_kernel_cpu")
    cpp, library = folder / "kernels.cpp", folder / "kernels.so"
    cpp.write_text(shim + source + "\n" + "\n".join(wrappers))
    result = subprocess.run([compiler, "-std=c++20", "-O2", "-shared", "-fPIC", "-pthread",
                             str(cpp), "-o", str(library)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    dll = ctypes.CDLL(str(library))

    def run(name, grid, *args):
        fn = getattr(dll, f"run_{name}")
        fn.argtypes = [ctypes.c_int] * 3 + [
            ctypes.c_void_p if "*" in p else ctypes.c_int if p.startswith("int ")
            else ctypes.c_double for p in declarations[name]
        ]
        fn.restype = None
        grid = (*grid, *(1 for _ in range(3 - len(grid))))
        fn(*grid, *(a.ctypes.data if isinstance(a, np.ndarray) else a for a in args))

    return run


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    return compile_kernel_source(tmp_path_factory, "_cuda_optimizer_kernels.cu")


@pytest.mark.parametrize("advance,hard", [(False, False), (True, False), (False, True)])
def test_gates_floor_and_cached_scores(kernel, advance, hard):
    rng = np.random.default_rng(301)
    n, d = 259, 3
    theta, v, u = rng.normal(size=d + 1), rng.normal(size=n), rng.normal(size=n)
    u[:2] = [-1000, 1000]
    original = u.copy()
    pi, q, sq = np.empty(n), np.empty((2, n)), np.empty((2, n))
    inv_r, alpha, temperature, floor = np.array([.8]), .3, .2, .07
    kernel("gates", (2, 1), n, d, temperature, floor, alpha, advance, hard,
           theta, v, inv_r, u, pi, q, sq)
    score = (original + alpha * v) * inv_r[0] if advance else original
    expected = ((score >= theta[-1]).astype(float) if hard else
                1 / (1 + np.exp(-np.clip((score - theta[-1]) / temperature, -50, 50))))
    weights = np.maximum([1 - expected, expected], 0 if hard else floor)
    np.testing.assert_allclose(u, score)
    np.testing.assert_allclose(pi, expected, atol=2e-16)
    np.testing.assert_allclose(q, weights, atol=2e-16)
    np.testing.assert_allclose(sq, np.sqrt(weights), atol=2e-16)


@pytest.mark.parametrize("ridge", [0., .03])
def test_augmented_weighted_qr_input_layout(kernel, ridge):
    rng = np.random.default_rng(302)
    n, p, side = 9, 4, 1
    m = n + (p if ridge else 0)
    phi, y = np.asfortranarray(rng.normal(size=(n, p))), rng.normal(size=n)
    sq = np.sqrt(rng.uniform(size=n))
    traces = np.array([1., np.sum((phi * sq[:, None]) ** 2)])
    a, b = np.empty((m, p), order="F"), np.empty(m)
    kernel("qr_inputs", (1, 1), n, p, m, side, ridge / p, phi, y, sq, traces, a, b)
    expected_a, expected_b = phi * sq[:, None], y * sq
    if ridge:
        expected_a = np.vstack([expected_a, np.sqrt(ridge * traces[side] / p) * np.eye(p)])
        expected_b = np.r_[expected_b, np.zeros(p)]
    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b)


def test_residual_gradient_and_tangent_direction(kernel):
    rng = np.random.default_rng(303)
    n, d, blocks = 259, 301, 2
    X, y = rng.normal(size=(n, d)), rng.normal(size=n)
    theta = rng.normal(size=d + 1)
    theta[:-1] /= np.linalg.norm(theta[:-1])
    pred = np.asfortranarray(rng.normal(size=(n, 2)))
    pi, temperature = rng.uniform(size=n), .7
    errors, coeff, partials = np.empty((2, n)), np.empty(n), np.empty((3, blocks))
    kernel("residual_stats", (blocks, 1), n, temperature, y, pred, pi, errors, coeff, partials)
    expected_errors = (y - pred.T) ** 2
    c = (expected_errors[1] - expected_errors[0]) * pi * (1 - pi) / (n * temperature)
    totals = partials.sum(axis=1)
    gw, gz = X.T @ c, -c.sum()
    expected_p = np.r_[-gw + (gw @ theta[:-1]) * theta[:-1], -gz]
    p, geometry, packet = np.empty(d + 1), np.empty(4), np.empty(7)
    traces, status = np.array([2., 3.]), np.zeros(2, dtype=np.int32)
    kernel("direction", (1, 1), d, .1, theta, gw, totals, traces, status, p, geometry, packet)
    np.testing.assert_allclose(errors, expected_errors)
    np.testing.assert_allclose(coeff, c, atol=1e-17)
    np.testing.assert_allclose(p, expected_p, atol=2e-16)
    np.testing.assert_allclose(geometry, [1, theta[:-1] @ p[:-1], p[:-1] @ p[:-1], -gz], atol=2e-16)
    expected_loss = np.mean((1 - pi) * expected_errors[0] + pi * expected_errors[1])
    np.testing.assert_allclose(packet, [expected_loss, np.linalg.norm(p), np.r_[gw, gz] @ p,
                                      np.count_nonzero(pi >= .5), .2, .3, 0], atol=2e-14)


def test_batched_cached_trials_and_commit_match_explicit_projection(kernel):
    rng = np.random.default_rng(304)
    n, d, blocks = 259, 3, 2
    X = rng.normal(size=(n, d))
    theta, p = rng.normal(size=d + 1), rng.normal(size=d + 1)
    theta[:-1] /= np.linalg.norm(theta[:-1])
    # Use a tangent direction while retaining the general w.p term in the formula.
    p[:-1] -= (p[:-1] @ theta[:-1]) * theta[:-1]
    u, v = X @ theta[:-1], X @ p[:-1]
    errors, temperature = rng.uniform(size=(2, n)), .4
    geometry = np.array([theta[:-1] @ theta[:-1], theta[:-1] @ p[:-1], p[:-1] @ p[:-1], p[-1]])
    alphas = np.array([8., 4., 2., 1., .5])
    partials = np.empty((3, blocks))
    kernel("trial_losses", (blocks, 3), n, d, 1, temperature, theta, u, v,
           errors, geometry, alphas, partials)
    expected = []
    for alpha in alphas[1:4]:
        candidate = theta + alpha * p
        candidate[:-1] /= np.linalg.norm(candidate[:-1])
        pi = 1 / (1 + np.exp(-np.clip((X @ candidate[:-1] - candidate[-1]) / temperature, -50, 50)))
        expected.append(np.mean((1 - pi) * errors[0] + pi * errors[1]))
    np.testing.assert_allclose(partials.sum(axis=1), expected, rtol=2e-15)
    inv_r = np.empty(1)
    original = theta.copy()
    kernel("commit_theta", (1, 1), d, alphas[2], p, geometry, theta, inv_r)
    candidate = original + alphas[2] * p
    candidate[:-1] /= np.linalg.norm(candidate[:-1])
    np.testing.assert_allclose(theta, candidate, atol=2e-16)
    np.testing.assert_allclose((u + alphas[2] * v) * inv_r[0], X @ theta[:-1], atol=2e-15)


@pytest.mark.parametrize("losses,selected", [([9., 8., 1.], 0), ([np.inf, 8., 1.], 1), ([np.inf, np.nan, 12.], -1)])
def test_armijo_selects_first_passing_step_not_minimum(kernel, losses, selected):
    losses, alphas = np.array(losses), np.array([1., .5, .25])
    packet, response = np.array([10., 1., -1., 0., 0., 0., 0.]), np.empty(3)
    kernel("select_trial", (1, 1), 3, 0, .1, losses, alphas, packet, response)
    k = selected if selected >= 0 else 2
    np.testing.assert_array_equal(response, [selected, losses[k], alphas[k]])


@pytest.mark.parametrize("problem", ["none", "zero", "tiny", "nonfinite", "info"])
def test_qr_safety_check_reports_failures(kernel, problem):
    a = np.asfortranarray(np.diag([1., 2., 3.]))
    b, info, status = np.ones(3), np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32)
    if problem == "zero": a[-1, -1] = 0.
    if problem == "tiny": a[-1, -1] = 1e-18
    if problem == "nonfinite": b[1] = np.nan
    if problem == "info": info[1] = -4
    kernel("qr_check", (1, 1), 3, 3, 1, 1e-15, a, b, info, status)
    np.testing.assert_array_equal(status, [0, problem != "none"])


def test_ridge_is_added_only_to_the_requested_gram_diagonal(kernel):
    gram = np.asfortranarray(np.arange(16., dtype=float).reshape(4, 4))
    expected = gram + .3 * 7. * np.eye(4)
    kernel("add_ridge", (1, 1), 4, 1, .3, np.array([5., 7.]), gram)
    np.testing.assert_allclose(gram, expected)


@pytest.mark.parametrize("problem", ["none", "zero_diagonal", "nonfinite", "factor_info", "solve_info"])
def test_cholesky_safety_check_reports_failures(kernel, problem):
    gram = np.asfortranarray(np.diag([1., 2., 3.]))
    coefficients = np.ones(3)
    info, status = np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32)
    if problem == "zero_diagonal": gram[1, 1] = 0.
    if problem == "nonfinite": coefficients[1] = np.nan
    if problem == "factor_info": info[0] = 2
    if problem == "solve_info": info[1] = -3
    kernel("cholesky_check", (1, 1), 3, 1, gram, coefficients, info, status)
    np.testing.assert_array_equal(status, [0, problem != "none"])
