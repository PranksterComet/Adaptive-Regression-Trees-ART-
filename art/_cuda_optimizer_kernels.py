"""Load shared single-node and batched CUDA optimizer kernels lazily."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def kernels(device_id: int):
    """Cache compiled functions per device; callers select the current device."""
    import cupy as cp

    names = ("gates", "qr_inputs", "qr_check", "residual_stats", "direction",
             "trial_losses", "select_trial", "commit_theta", "add_ridge", "cholesky_check",
             "batched_gates", "batched_residual_stats", "batched_direction",
             "batched_trial_losses", "batched_trial_reduce", "batched_select_trial",
             "batched_commit_theta")
    module = cp.RawModule(
        code=Path(__file__).with_name("_cuda_optimizer_kernels.cu").read_text(),
        options=("--std=c++11",),
        name_expressions=names,
    )
    return {name: module.get_function(name) for name in names}
