"""Check backend selection and paired benchmark outputs without requiring CUDA."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    import art._cuda_cholesky as cholesky

    monkeypatch.setattr(cholesky, "cuda_cholesky_status", lambda: (True, "mock Cholesky"))
    path = Path(__file__).with_name("splitter_performance.py")
    spec = importlib.util.spec_from_file_location("splitter_performance_cuda_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_backend_selection_is_explicit_and_cpu_does_not_probe(benchmark, monkeypatch):
    import art._cuda_hybrid_gram as cuda
    import art._cuda_qr as qr

    def unexpected_probe():
        raise AssertionError("Explicit CPU mode must not probe CUDA")

    monkeypatch.setattr(cuda, "cuda_status", unexpected_probe)
    monkeypatch.setattr(qr, "cuda_qr_status", unexpected_probe)
    assert benchmark.select_backends("cpu", "auto")[0] == ["cpu"]
    monkeypatch.setattr(cuda, "cuda_status", lambda: (False, "no CUDA driver"))
    assert benchmark.select_backends("auto", "auto") == (["cpu"], "no CUDA driver")
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        benchmark.select_backends("both", "auto")
    monkeypatch.setattr(cuda, "cuda_status", lambda: (True, "test GPU"))
    monkeypatch.setattr(qr, "cuda_qr_status", lambda: (True, "test QR"))
    assert benchmark.select_backends("auto", "auto")[0] == ["cpu", "hybrid_gram", "cupy_qr", "cupy_cholesky"]
    assert benchmark.select_backends("cuda", "auto")[0] == ["hybrid_gram"]
    assert benchmark.select_backends("auto", "qr")[0] == ["cpu", "cupy_qr", "cupy_cholesky"]


def test_default_arguments(benchmark, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["splitter_performance.py"])
    args = benchmark.parse_args()
    assert args.backend == "auto"
    assert args.dimensions == [5, 10, 20, 40]
    assert args.sample_multiplier == 5
    assert args.cpu_threads == 16
    assert args.min_side_points == 1 and args.min_side_fraction == .05
    assert args.backtrack_batch_size == 8 and args.Xw_refresh == 32
    assert args.output_dir == benchmark.PROJECT_ROOT / "tests" / "splitter_performance_outputs"


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_cpu_threads_must_be_positive(benchmark, monkeypatch, capsys, threads):
    monkeypatch.setattr(sys, "argv", ["splitter_performance.py", "--cpu-threads", threads])
    with pytest.raises(SystemExit) as caught:
        benchmark.parse_args()
    assert caught.value.code == 2
    assert "--cpu-threads must be positive" in capsys.readouterr().err


def test_auto_without_cuda_generates_cpu_reports(benchmark, monkeypatch, tmp_path):
    import art._cuda_hybrid_gram as cuda
    import art._cuda_qr as qr

    monkeypatch.setattr(cuda, "cuda_status", lambda: (False, "no CUDA driver"))

    def unexpected_probe():
        raise AssertionError("QR need not be probed when CUDA is unavailable")

    monkeypatch.setattr(qr, "cuda_qr_status", unexpected_probe)
    monkeypatch.setattr(sys, "argv", [
        "splitter_performance.py", "--dimensions", "2", "--trials", "1",
        "--probe-multiplier", "2", "--max-iters", "0", "--output-dir", str(tmp_path),
    ])
    benchmark.main()
    assert (tmp_path / "report.txt").exists()
    assert not (tmp_path / "report_cuda.txt").exists()
    assert not (tmp_path / "report_cuda_qr.txt").exists()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["backends"] == ["cpu"]
    assert metadata["cuda"] == "no CUDA driver"
    with (tmp_path / "timings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(int(row["n_samples"]) == 5 * int(row["effective_dimension"]) for row in rows)


def test_qr_backend_probes_qr_and_can_compare_all_backends(benchmark, monkeypatch):
    import art._cuda_qr as qr
    import art._cuda_hybrid_gram as hybrid

    monkeypatch.setattr(qr, "cuda_qr_status", lambda: (True, "mock QR"))
    monkeypatch.setattr(hybrid, "cuda_status", lambda: (True, "mock BLAS"))
    assert benchmark.select_backends("cuda-qr", "svd")[0] == ["cupy_qr"]
    assert benchmark.select_backends("all", "auto")[0] == ["cpu", "hybrid_gram", "cupy_qr", "cupy_cholesky"]
    assert benchmark.select_backends("all", "qr")[0] == ["cpu", "cupy_qr", "cupy_cholesky"]
    monkeypatch.setattr(qr, "cuda_qr_status", lambda: (False, "missing cuSOLVER"))
    backends, status = benchmark.select_backends("auto", "auto")
    assert backends == ["cpu", "hybrid_gram", "cupy_cholesky"]
    assert "QR unavailable: missing cuSOLVER" in status
    assert benchmark.select_backends("auto", "qr")[0] == ["cpu", "cupy_cholesky"]
    with pytest.raises(RuntimeError, match="CUDA QR requested but unavailable"):
        benchmark.select_backends("cuda-qr", "auto")
    with pytest.raises(RuntimeError, match="CUDA QR requested but unavailable"):
        benchmark.select_backends("all", "auto")


@pytest.mark.parametrize("thread_args,expected_threads", [([], 16), (["--cpu-threads", "3"], 3)])
def test_both_reports_share_cases_and_seeds_and_apply_cpu_thread_limit(
    benchmark, monkeypatch, tmp_path, thread_args, expected_threads
):
    import scipy.linalg
    from threadpoolctl import threadpool_info

    before_pools = threadpool_info()
    monkeypatch.setattr(benchmark, "select_backends", lambda *args: (["cpu", "hybrid_gram"], "mock GPU"))
    actual_run_trial = benchmark.run_trial
    seen = {"cpu": [], "hybrid_gram": []}

    def cpu_replay(case, args, random_state, optimizer="cpu"):
        # This test exercises report orchestration, not CUDA numerical kernels.
        assert all(pool["num_threads"] == expected_threads
                   for pool in threadpool_info() if pool["user_api"] == "blas")
        if optimizer == "hybrid_gram":
            assert (tmp_path / "report.txt").exists()
            assert (tmp_path / "timings.csv").exists()
        seen[optimizer].append((id(case), random_state))
        row = actual_run_trial(case, args, random_state, optimizer="cpu")
        row["optimizer"] = optimizer
        return row

    monkeypatch.setattr(benchmark, "run_trial", cpu_replay)
    monkeypatch.setattr(sys, "argv", [
        "splitter_performance.py", "--dimensions", "2", "--trials", "2",
        "--sample-multiplier", "5", "--probe-multiplier", "2",
        "--max-iters", "1", "--output-dir", str(tmp_path),
        *thread_args,
    ])
    import os
    before = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
    )}
    benchmark.main()
    assert threadpool_info() == before_pools
    assert seen["cpu"] == seen["hybrid_gram"]
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["thread_environment"] == before
    assert metadata["cpu_threads"] == expected_threads
    assert all(pool["num_threads"] == expected_threads
               for pool in metadata["blas_threadpools"] if pool["user_api"] == "blas")
    for suffix, backend in (("", "cpu"), ("_cuda", "hybrid_gram")):
        report = (tmp_path / f"report{suffix}.txt").read_text()
        assert report.count("[case:") == 4
        assert f"optimizer: {backend}" in report
        assert f"cpu_threads: {expected_threads} (BLAS limit)" in report
        with (tmp_path / f"timings{suffix}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 8
        assert {row["optimizer"] for row in rows} == {backend}
        for row in rows:
            total = float(row["total_seconds"])
            least_squares = float(row["least_squares_seconds"])
            assert least_squares <= total
            assert float(row["least_squares_percent"]) == pytest.approx(
                100 * least_squares / total
            )


def test_auto_generates_all_four_reports_with_same_data_and_seeds(benchmark, monkeypatch, tmp_path):
    import art._cuda_hybrid_gram as hybrid
    import art._cuda_qr as qr

    monkeypatch.setattr(hybrid, "cuda_status", lambda: (True, "mock BLAS"))
    monkeypatch.setattr(qr, "cuda_qr_status", lambda: (True, "mock QR"))
    actual_run = benchmark.run_trial
    seen = {"cpu": [], "hybrid_gram": [], "cupy_qr": [], "cupy_cholesky": []}

    def replay(case, args, random_state, optimizer="cpu"):
        from threadpoolctl import threadpool_info

        assert all(pool["num_threads"] == 16
                   for pool in threadpool_info() if pool["user_api"] == "blas")
        seen[optimizer].append((id(case), random_state))
        if optimizer != "cpu":
            assert (tmp_path / "report.txt").exists()
        row = actual_run(case, args, random_state, optimizer="cpu")
        row["backend"] = optimizer
        row.update(benchmark.backend_metadata(optimizer))
        return row

    monkeypatch.setattr(benchmark, "run_trial", replay)
    monkeypatch.setattr(sys, "argv", [
        "splitter_performance.py", "--dimensions", "2", "--trials", "1",
        "--sample-multiplier", "5", "--probe-multiplier", "2", "--max-iters", "0",
        "--backtrack-batch-size", "3", "--qr-serial", "--output-dir", str(tmp_path),
    ])
    benchmark.main()
    assert seen["cpu"] == seen["hybrid_gram"] == seen["cupy_qr"] == seen["cupy_cholesky"]
    for suffix in ("", "_cuda", "_cuda_qr", "_cuda_cholesky"):
        assert (tmp_path / f"report{suffix}.txt").exists()
        assert (tmp_path / f"timings{suffix}.csv").exists()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["backends"] == ["cpu", "hybrid_gram", "cupy_qr", "cupy_cholesky"]
    assert metadata["cpu_threads"] == 16
    report = (tmp_path / "report_cuda_qr.txt").read_text()
    assert "optimizer: cupy_qr" in report
    assert "backtrack_batch_size: 3" in report
    assert "qr_parallel: False" in report
    assert "speculative" in report
    with (tmp_path / "timings_cuda_qr.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["backend"] for row in rows} == {"cupy_qr"}
    assert {row["optimizer"] for row in rows} == {"cupy_qr"}
    chol = (tmp_path / "report_cuda_cholesky.txt").read_text()
    assert "optimizer: cupy_cholesky" in chol
    assert "gpu_solver_requested: normal" in chol
    assert "solver_fallback: false" in chol
    assert "complementary_grams: true" in chol
    assert "gram_floor_correction: true" in chol
    assert metadata["backend_settings"]["cupy_cholesky"]["gpu_solver"] == "cholesky"
    with (tmp_path / "timings_cuda_cholesky.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["optimizer"] for row in rows} == {"cupy_cholesky"}
    assert {row["gpu_solver"] for row in rows} == {"cholesky"}
    assert {row["solver_fallback"] for row in rows} == {"False"}


def test_cholesky_selection_is_independent_of_qr_and_reports_probe_failures(benchmark, monkeypatch):
    import art._cuda_cholesky as cholesky
    import art._cuda_qr as qr
    import art._cuda_hybrid_gram as hybrid

    def unexpected():
        raise AssertionError("Cholesky-only mode must not probe another backend")

    monkeypatch.setattr(qr, "cuda_qr_status", unexpected)
    monkeypatch.setattr(hybrid, "cuda_status", unexpected)
    for mode in ("cuda_cholesky", "cuda-cholesky"):
        assert benchmark.select_backends(mode, "svd")[0] == ["cupy_cholesky"]
    monkeypatch.setattr(cholesky, "cuda_cholesky_status", lambda: (False, "broken Cholesky"))
    with pytest.raises(RuntimeError, match="CUDA Cholesky requested but unavailable"):
        benchmark.select_backends("cuda_cholesky", "normal")
    monkeypatch.setattr(qr, "cuda_qr_status", lambda: (True, "mock QR"))
    monkeypatch.setattr(hybrid, "cuda_status", lambda: (True, "mock GPU"))
    backends, status = benchmark.select_backends("auto", "auto")
    assert backends == ["cpu", "hybrid_gram", "cupy_qr"]
    assert "Cholesky unavailable: broken Cholesky" in status
    with pytest.raises(RuntimeError, match="CUDA Cholesky requested but unavailable"):
        benchmark.select_backends("all", "auto")
