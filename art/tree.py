"""Tree data structures and prediction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union

import numpy as np

from .domain import PolytopeRegion
from .models import RegressionModel


@dataclass
class LeafNode:
    model: RegressionModel
    region: Optional[PolytopeRegion] = None
    depth: int = 0
    node_id: str = "root"
    status: str = "leaf"
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_one(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.model.predict(x)[0])


TreeNode = Union["SplitNode", LeafNode]


@dataclass
class SplitNode:
    w: np.ndarray
    z: float
    left: TreeNode
    right: TreeNode
    region: Optional[PolytopeRegion] = None
    depth: int = 0
    node_id: str = "root"
    status: str = "split"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float).reshape(-1)
        self.z = float(self.z)

    def route(self, x: np.ndarray) -> TreeNode:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.w.shape[0]:
            raise ValueError(f"x must have shape ({self.w.shape[0]},), got {x.shape}.")
        return self.right if float(self.w @ x - self.z) >= 0.0 else self.left


class RegressionTree:
    """Prediction-only regression tree."""

    def __init__(
        self,
        root: Optional[TreeNode] = None,
        oracle_queries: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.root = root
        self.oracle_queries = int(oracle_queries)
        self.metadata = {} if metadata is None else dict(metadata)

    def predict_one(self, x: np.ndarray) -> float:
        leaf = self.leaf_for_point(x)
        return leaf.predict_one(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        return np.array([self.predict_one(x) for x in X], dtype=float)

    def leaf_for_point(self, x: np.ndarray) -> LeafNode:
        if self.root is None:
            raise ValueError("Tree has no root node.")

        node = self.root
        while isinstance(node, SplitNode):
            node = node.route(x)
        return node

    def iter_nodes(self) -> Iterator[TreeNode]:
        """Yield all nodes in depth-first, left-to-right order."""

        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, SplitNode):
                stack.extend([node.right, node.left])

    def iter_leaves(self) -> Iterator[LeafNode]:
        """Yield all leaf nodes in depth-first, left-to-right order."""

        for node in self.iter_nodes():
            if isinstance(node, LeafNode):
                yield node

    def num_nodes(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    def num_split_nodes(self) -> int:
        return sum(isinstance(node, SplitNode) for node in self.iter_nodes())

    def num_leaves(self) -> int:
        return sum(1 for _ in self.iter_leaves())

    def max_depth(self) -> int:
        def depth(node: TreeNode) -> int:
            if isinstance(node, LeafNode):
                return node.depth
            return max(depth(node.left), depth(node.right))

        return 0 if self.root is None else depth(self.root)
