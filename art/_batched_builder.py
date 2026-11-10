"""FIFO node scheduling; numerical optimization and tree policies live elsewhere.

The default dispatcher runs the builder's existing scalar splitter. split_batch
connects the resident GPU adapter and CUDA-free tests:
it receives prepared work records and must return final tree nodes in the same
order, using builder._finish_node to preserve acceptance/diagnostic policy.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable

from .tree import SplitNode, TreeNode

if TYPE_CHECKING:
    from .builder import RegressionTreeBuilder, _NodeWork


def build_bfs(
    builder: RegressionTreeBuilder,
    work: _NodeWork,
    split_batch: Callable[[list[_NodeWork]], list[TreeNode]] | None = None,
) -> TreeNode:
    """Process oldest jobs first, collecting at most node_batch_size candidates.

    Terminal nodes are attached immediately and do not occupy a ready slot.
    Children retain inherited CPU samples until dequeued. Cohorts may straddle
    levels; their results are attached in input order, appending left then right.
    No GPU storage is allocated or owned by the queue.
    """
    pending = deque([work])
    root = None

    def attach(job, node):
        nonlocal root
        if job.parent is None:
            root = node
        else:
            setattr(job.parent, job.side, node)

    while pending:
        ready = []
        while pending and len(ready) < builder.node_batch_size:
            job = pending.popleft()
            leaf = builder._prepare_node(job)
            if leaf is None:
                ready.append(job)
            else:
                attach(job, leaf)
        if not ready:
            continue
        nodes = ([builder._split_node(job) for job in ready] if split_batch is None
                 else split_batch(ready))
        if len(nodes) != len(ready):
            raise RuntimeError("Split dispatch must return one node per ready job.")
        for job, node in zip(ready, nodes):
            attach(job, node)
            if isinstance(node, SplitNode):
                pending.extend(builder._children(job, node))
    assert root is not None  # The nonempty queue always starts with its root job.
    return root
