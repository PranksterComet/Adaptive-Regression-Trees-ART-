"""Check benchmark scheduling/reporting on CPU and numerical paths on CUDA."""

from contextlib import nullcontext
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "tacc_scripts" / "cupy" / "cupy_lstsq_batching.py"
    spec = importlib.util.spec_from_file_location("cupy_lstsq_batching_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_and_feature_counts(benchmark):
    args = benchmark.parse_args([])
    assert args.dimensions == [5, 10, 20, 40]
    assert (args.sample_multiplier, args.batch_size, args.cpu_threads) == (10, 20, 20)
    for d, linear, quadratic in ((5, 6, 21), (10, 11, 66), (20, 21, 231), (40, 41, 861)):
        assert benchmark.problem_sizes(d, "linear", 10) == (10 * linear, linear)
        assert benchmark.problem_sizes(d, "quadratic", 10) == (10 * quadratic, quadratic)


@pytest.mark.parametrize("flag,value", [
    ("--dimensions", "0"), ("--sample-multiplier", "0"), ("--batch-size", "0"),
    ("--cpu-threads", "0"), ("--repeats", "0"), ("--warmup", "-1"), ("--seed", "-1"),
])
def test_invalid_arguments(benchmark, flag, value):
    with pytest.raises(SystemExit):
        benchmark.parse_args([flag, value])


def test_data_are_reproducible_and_target_length_is_n(benchmark):
    args = benchmark.parse_args(["--dimensions", "2", "-s", "3", "-K", "3"])
    phi, y = benchmark.make_problems(2, "quadratic", args)
    again_phi, again_y = benchmark.make_problems(2, "quadratic", args)
    assert phi.shape == (3, 18, 6)
    assert y.shape == (3, 18, 1)
    assert phi.dtype == y.dtype == np.float64
    np.testing.assert_array_equal(phi, again_phi)
    np.testing.assert_array_equal(y, again_y)
    assert not np.array_equal(phi[0], phi[1])


@pytest.fixture
def fake_cupy():
    """NumPy arithmetic with a stream log; this does not emulate GPU execution."""
    log, current = [], []

    class Stream:
        def __init__(self, **kwargs):
            self.id = id(self)

        def __enter__(self):
            current.append(self)
            return self

        def __exit__(self, *args):
            current.pop()

        def synchronize(self):
            log.append(("sync", self.id))

    class Event:
        clock = 0

        def record(self, stream):
            Event.clock += 1
            self.time = Event.clock

    def qr(phi, mode):
        log.append(("qr", current[-1].id, phi.shape))
        return np.linalg.qr(phi, mode=mode)

    def chol(gram):
        log.append(("chol", current[-1].id, gram.shape))
        return np.linalg.cholesky(gram)

    runtime = SimpleNamespace(getDeviceProperties=lambda device: {"name": b"test GPU"},
                              runtimeGetVersion=lambda: 12000, driverGetVersion=lambda: 12000)
    cuda = SimpleNamespace(Stream=Stream, Event=Event, runtime=runtime,
                           Device=lambda: SimpleNamespace(id=0),
                           get_elapsed_time=lambda start, end: float(end.time - start.time))
    cp = SimpleNamespace(cuda=cuda, asarray=np.asarray, asnumpy=np.asarray, matmul=np.matmul,
                         linalg=SimpleNamespace(qr=qr, cholesky=chol), __version__="14.2.0-test",
                         get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None))

    def triangular(a, b, lower=False, trans="N", **kwargs):
        return np.linalg.solve(a.swapaxes(-1, -2) if trans == "T" else a, b)

    return cp, triangular, log


@pytest.mark.parametrize("solver", ["qr", "cholesky"])
def test_concurrent_enqueues_every_problem_before_waiting(benchmark, fake_cupy, solver):
    cp, triangular, log = fake_cupy
    rng = np.random.default_rng(501)
    phi, y = rng.normal(size=(3, 18, 6)), rng.normal(size=(3, 18, 1))
    problems = benchmark.GpuProblems(cp, phi, y, triangular)
    log.clear()
    result = problems.download(problems.run(solver, "concurrent"))
    factors = [i for i, item in enumerate(log) if item[0] in ("qr", "chol")]
    waits = [i for i, item in enumerate(log) if item[0] == "sync"]
    assert len(factors) == len(waits) == 3
    assert max(factors) < min(waits)
    assert len({log[i][1] for i in factors}) == 3
    for j in range(3):
        expected = np.linalg.lstsq(phi[j], y[j], rcond=None)[0]
        np.testing.assert_allclose(result[j], expected, atol=1e-12)

    log.clear()
    problems.run(solver, "sequential")
    assert [item[0] for item in log] == ["qr" if solver == "qr" else "chol", "sync"] * 3


@pytest.mark.parametrize("solver", ["qr", "cholesky"])
@pytest.mark.parametrize("mode", ["sequential", "concurrent", "batched"])
def test_measurement_sums_phases_and_omits_concurrent_breakdown(benchmark, fake_cupy, solver, mode):
    cp, triangular, _ = fake_cupy
    rng = np.random.default_rng(502)
    problems = benchmark.GpuProblems(cp, rng.normal(size=(3, 18, 6)),
                                     rng.normal(size=(3, 18, 1)), triangular)
    times, coefficients = benchmark.measure(problems, solver, mode, warmup=1, repeats=2)
    assert coefficients.shape == (3, 6, 1)
    assert times["total_ms"] > 0
    for name in benchmark.PHASES[solver]:
        expected = None if mode == "concurrent" else 3. if mode == "sequential" else 1.
        assert times[f"{name}_ms"] == expected
    if solver == "qr":
        assert times["gram_rhs_ms"] is None


def test_complete_report_and_csv_with_cpu_arithmetic_adapter(benchmark, fake_cupy, monkeypatch, tmp_path):
    cp, triangular, _ = fake_cupy
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupyx", SimpleNamespace(errstate=lambda **kwargs: nullcontext()))
    monkeypatch.setitem(sys.modules, "cupyx.scipy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", SimpleNamespace(solve_triangular=triangular))
    benchmark.main(["--dimensions", "2", "-s", "3", "-K", "3", "--warmup", "0",
                    "--repeats", "2", "--output-dir", str(tmp_path)])
    report = (tmp_path / "report.txt").read_text()
    assert "NOT native batched QR" in report
    assert "[total wall time" in report
    assert "[qr phase times" in report
    assert "[cholesky phase times" in report
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["complete"]
    assert metadata["arguments"]["cpu_threads"] == 20
    with (tmp_path / "timings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12  # Two models, two solvers, three modes.
    for row in rows:
        assert float(row["coefficient_relative_error"]) < 1e-10
        if row["mode"] == "concurrent":
            assert row["factor_ms"] == row["solve_ms"] == row["gram_rhs_ms"] == ""


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if not count:
        pytest.skip("CUDA unavailable")
    return cp


@pytest.mark.parametrize("solver", ["qr", "cholesky"])
@pytest.mark.parametrize("mode", ["sequential", "concurrent", "batched"])
def test_real_gpu_solves_and_phase_timings(benchmark, gpu, solver, mode):
    import cupyx
    from cupyx.scipy.linalg import solve_triangular

    rng = np.random.default_rng(503)
    phi, y = rng.normal(size=(3, 30, 6)), rng.normal(size=(3, 30, 1))
    problems = benchmark.GpuProblems(gpu, phi, y, solve_triangular)
    try:
        with cupyx.errstate(linalg="ignore"):
            times, coefficients = benchmark.measure(problems, solver, mode, warmup=1, repeats=2)
        for j in range(3):
            expected = np.linalg.lstsq(phi[j], y[j], rcond=None)[0]
            np.testing.assert_allclose(coefficients[j], expected, rtol=1e-10, atol=1e-12)
        if mode != "concurrent":
            assert all(times[f"{name}_ms"] >= 0 for name in benchmark.PHASES[solver])
    finally:
        problems.synchronize()
