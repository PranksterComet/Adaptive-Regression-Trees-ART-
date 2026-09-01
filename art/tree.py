"""Tree data structures and prediction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Union

import numpy as np

from .domain import PolytopeRegion
from .models import RegressionModel


TREE_ARTIFACT_TYPE = "art.regression_tree"
TREE_ARTIFACT_VERSION = 1


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

    def save(
        self,
        path: str | Path,
        run_config: Mapping[str, Any] | None = None,
        compress: int = 3,
    ) -> Path:
        """Save the fitted tree, diagnostics, and optional run configuration."""

        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_type": TREE_ARTIFACT_TYPE,
            "format_version": TREE_ARTIFACT_VERSION,
            "tree": self,
            "run_config": {} if run_config is None else dict(run_config),
        }
        joblib.dump(artifact, path, compress=compress)
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["RegressionTree", dict[str, Any]]:
        """Load a trusted tree artifact and its saved run configuration."""

        import joblib

        path = Path(path)
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise TypeError(f"{path} is not a regression-tree artifact.")
        if artifact.get("artifact_type") != TREE_ARTIFACT_TYPE:
            raise ValueError(f"{path} has an unrecognized artifact type.")
        version = artifact.get("format_version")
        if version != TREE_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported tree artifact version {version!r}; "
                f"expected {TREE_ARTIFACT_VERSION}."
            )

        tree = artifact.get("tree")
        if not isinstance(tree, cls):
            raise TypeError(f"{path} does not contain a RegressionTree.")
        run_config = artifact.get("run_config", {})
        if not isinstance(run_config, dict):
            raise TypeError(f"{path} contains an invalid run configuration.")
        return tree, run_config

    def predict_one(self, x: np.ndarray) -> float:
        leaf = self.leaf_for_point(x)
        return leaf.predict_one(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {X.shape}.")
        if X.shape[0] == 0:
            return np.empty(0, dtype=float)
        if self.root is None:
            raise ValueError("Tree has no root node.")

        predictions = np.empty(X.shape[0], dtype=float)
        pending: list[tuple[TreeNode, np.ndarray]] = [
            (self.root, np.arange(X.shape[0]))
        ]

        while pending:
            node, indices = pending.pop()
            if isinstance(node, LeafNode):
                leaf_predictions = np.asarray(
                    node.model.predict(X[indices]),
                    dtype=float,
                ).reshape(-1)
                if leaf_predictions.shape[0] != indices.shape[0]:
                    raise ValueError(
                        f"Leaf model returned {leaf_predictions.shape[0]} predictions "
                        f"for {indices.shape[0]} samples."
                    )
                predictions[indices] = leaf_predictions
                continue

            if X.shape[1] != node.w.shape[0]:
                raise ValueError(
                    f"X must have {node.w.shape[0]} columns, got {X.shape[1]}."
                )
            right_mask = X[indices] @ node.w - node.z >= 0.0
            if np.any(~right_mask):
                pending.append((node.left, indices[~right_mask]))
            if np.any(right_mask):
                pending.append((node.right, indices[right_mask]))

        return predictions

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

    def path_to_node(self, node_id: str) -> tuple[TreeNode, ...]:
        """Return the root-to-node path for a path-encoded identifier."""

        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a nonempty string.")
        if self.root is None:
            raise KeyError(f"Tree does not contain a node with id {node_id!r}.")
        if node_id == self.root.node_id:
            return (self.root,)

        prefix = f"{self.root.node_id}/"
        if not node_id.startswith(prefix):
            raise KeyError(f"Tree does not contain a node with id {node_id!r}.")

        path: list[TreeNode] = [self.root]
        node = self.root
        for branch in node_id[len(prefix) :].split("/"):
            if branch not in ("L", "R") or not isinstance(node, SplitNode):
                raise KeyError(f"Tree does not contain a node with id {node_id!r}.")
            node = node.left if branch == "L" else node.right
            path.append(node)

        if node.node_id != node_id:
            raise KeyError(f"Tree does not contain a node with id {node_id!r}.")
        return tuple(path)

    def get_node(self, node_id: str) -> TreeNode:
        """Return the node with the requested identifier."""

        return self.path_to_node(node_id)[-1]

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
