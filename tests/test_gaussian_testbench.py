from __future__ import annotations

import numpy as np

from art.tree import LeafNode, RegressionTree, SplitNode
from examples.gaussian_testbench import (
    alignment_rows,
    make_target,
    parse_args,
    split_alignment_by_depth,
    validate_args,
)


class ZeroModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


def test_gaussian_testbench_builds_seeded_target() -> None:
    args = parse_args(["--dimension", "6", "--active-dimension", "4", "--seed", "13"])
    validate_args(args)
    first, _, metadata = make_target(args)
    second, _, _ = make_target(args)

    np.testing.assert_allclose(first.tau, [0.2, 0.5, 0.8, 1.1])
    np.testing.assert_array_equal(first.U, second.U)
    assert metadata["inactive_dimension"] == 2
    assert metadata["rotation_distribution"] == "Haar"


def test_gaussian_testbench_exposes_min_side_fallback() -> None:
    args = parse_args(["--min-side-fallback"])

    assert args.min_side_fallback is True


def test_gaussian_testbench_exposes_parallel_sampler() -> None:
    args = parse_args(["--sampler", "parallel_numba"])

    assert args.sampler == "parallel_numba"
    assert args.n_chains == 50
    assert args.sampler_threads == 10
    assert args.parallel_random


def test_split_alignment_is_grouped_by_depth() -> None:
    leaf = lambda node_id, depth: LeafNode(ZeroModel(), node_id=node_id, depth=depth)
    right = SplitNode(
        w=np.array([0.0, 0.0, 1.0]),
        z=0.0,
        left=leaf("root/R/L", 2),
        right=leaf("root/R/R", 2),
        node_id="root/R",
        depth=1,
    )
    root = SplitNode(
        w=np.array([1.0, 0.0, 0.0]),
        z=0.0,
        left=leaf("root/L", 1),
        right=right,
        node_id="root",
        depth=0,
    )
    tree = RegressionTree(root)
    U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    alignments = split_alignment_by_depth(tree, U)
    rows = alignment_rows(alignments)

    np.testing.assert_allclose(alignments[0], [1.0])
    np.testing.assert_allclose(alignments[1], [0.0])
    assert rows[0]["depth"] == 0
    assert rows[0]["mean"] == 1.0
    assert rows[1]["mean"] == 0.0
