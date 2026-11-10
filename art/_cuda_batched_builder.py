"""Connect FIFO cohorts to one resident GPU optimizer, retaining builder policy."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ._cuda_batched_optimizer import BatchedOptimizerWorkspace
from .splitters import SplitNotFoundError

if TYPE_CHECKING:
    from .builder import RegressionTreeBuilder, _NodeWork
    from .tree import TreeNode


class BatchedSplitDispatcher:
    """Own one capacity workspace for a BFS build; return finalized CPU nodes.

    Temperature selection, acceptance and retries use builder helpers. Tuning
    candidates still use the existing scalar splitter. Cohort data is loaded
    once; retries select resident slots, with fallback enabled only on a node's
    final allowed attempt. The usual setting is one additional builder retry.
    """

    def __init__(self, builder: RegressionTreeBuilder):
        self.builder = builder
        splitter = copy.deepcopy(builder.splitter)
        # Seed once from the build RNG; never mutate the configured splitter.
        splitter.random_state = builder._next_seed()
        with builder._timing_context("splitter"):
            self.ws = BatchedOptimizerWorkspace.allocate(
                splitter, builder.node_batch_size, builder.target_samples,
                builder.dimension, reorder=builder.batch_reorder,
            )

    def __call__(self, ready: list[_NodeWork]) -> list[TreeNode]:
        b, ws = self.builder, self.ws
        nodes = [None] * len(ready)
        jobs, states, positions = [], [], []
        for pos, job in enumerate(ready):
            try:
                state = b._split_state(job.samples.X, job.samples.y, job.depth, job.prepared)
            except SplitNotFoundError as exc:
                nodes[pos] = b._finish_node(job, exc)
                continue
            jobs.append(job)
            states.append(state)
            positions.append(pos)
            # Parent fitting is complete. Retain CPU Phi only if a rejected
            # validation fit would need a full-data refit when becoming a leaf.
            if not b.validation_fraction:
                job.prepared = None
        if not jobs:
            return nodes

        k = len(jobs)
        parent_loss = np.array([job.parent_mse for job in jobs])
        temperature = np.array([ws.s.temperature if s.temperature is None else s.temperature
                                for s in states])
        outcomes = [None] * k
        with b._timing_context("splitter"):
            for j, job in enumerate(jobs):
                ws.host_X[j] = job.samples.X
                ws.host_y[j] = job.samples.y
            ws.load(ws.host_X[:k], ws.host_y[:k], temperature=temperature)
            pending = list(range(k))
            while pending:
                again = []
                # Mixed per-node retry limits can require different fallback
                # settings. Normally every pending slot shares one setting.
                groups = [[j for j in pending if states[j].fallback == fallback]
                          for fallback in (False, True)]
                for fallback, ids in zip((False, True), groups):
                    if not ids:
                        continue
                    results = ws.optimize(
                        nodes=ids, parent_loss=parent_loss[ids],
                        min_side_fallback=fallback,
                        hard_refit=b.hard_refits,
                    )  # Omit theta0: initialize w and z entirely on GPU.
                    for j, result in zip(ids, results):
                        if b._retry_split(states[j], result):
                            again.append(j)
                        else:
                            outcomes[j] = result
                pending = sorted(again)

        for pos, job, state, result in zip(positions, jobs, states, outcomes):
            if isinstance(result, SplitNotFoundError):
                nodes[pos] = b._finish_node(job, state.error(result))
            else:
                nodes[pos] = b._finish_node(job, result, {**state.context, **state.metadata})
        return nodes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.ws.close()
