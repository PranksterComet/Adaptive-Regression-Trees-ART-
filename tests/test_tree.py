from __future__ import annotations

import numpy as np

from art.tree import LeafNode, RegressionTree, SplitNode


class CountingModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.predict_calls = 0
        self.batch_sizes: list[int] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.predict_calls += 1
        self.batch_sizes.append(X.shape[0])
        return np.full(X.shape[0], self.value)


def make_test_tree() -> tuple[RegressionTree, tuple[CountingModel, ...]]:
    left_model = CountingModel(-1.0)
    lower_right_model = CountingModel(2.0)
    upper_right_model = CountingModel(3.0)

    right = SplitNode(
        w=np.array([0.0, 1.0]),
        z=0.0,
        left=LeafNode(lower_right_model, node_id="root/R/L"),
        right=LeafNode(upper_right_model, node_id="root/R/R"),
        node_id="root/R",
    )
    root = SplitNode(
        w=np.array([1.0, 0.0]),
        z=0.0,
        left=LeafNode(left_model, node_id="root/L"),
        right=right,
        node_id="root",
    )
    return RegressionTree(root), (left_model, lower_right_model, upper_right_model)


def test_predict_routes_batches_and_preserves_order() -> None:
    tree, models = make_test_tree()
    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, 4.0],
            [2.0, -1.0],
            [-2.0, -3.0],
            [0.0, 0.0],
        ]
    )

    predictions = tree.predict(X)

    np.testing.assert_array_equal(predictions, np.array([3.0, -1.0, 2.0, -1.0, 3.0]))
    assert [model.predict_calls for model in models] == [1, 1, 1]
    assert [model.batch_sizes for model in models] == [[2], [1], [2]]


def test_predict_skips_leaves_with_no_routed_samples() -> None:
    tree, models = make_test_tree()
    X = np.array([[-1.0, 1.0], [-2.0, -1.0]])

    np.testing.assert_array_equal(tree.predict(X), np.array([-1.0, -1.0]))
    assert [model.predict_calls for model in models] == [1, 0, 0]


def test_predict_accepts_one_dimensional_input() -> None:
    tree, _ = make_test_tree()
    x = np.array([1.0, -1.0])

    prediction = tree.predict(x)

    assert prediction.shape == (1,)
    np.testing.assert_array_equal(prediction, np.array([tree.predict_one(x)]))


def test_get_node_finds_split_and_leaf_nodes() -> None:
    tree, _ = make_test_tree()

    assert tree.get_node("root") is tree.root
    assert tree.get_node("root/R") is tree.root.right
    assert tree.get_node("root/R/L") is tree.root.right.left


def test_path_to_node_returns_root_through_requested_node() -> None:
    tree, _ = make_test_tree()

    assert tree.path_to_node("root") == (tree.root,)
    assert tree.path_to_node("root/R/L") == (
        tree.root,
        tree.root.right,
        tree.root.right.left,
    )


def test_get_node_rejects_invalid_or_unknown_ids() -> None:
    tree, _ = make_test_tree()

    with np.testing.assert_raises(ValueError):
        tree.get_node("")
    with np.testing.assert_raises(KeyError):
        tree.get_node("root/missing")
    with np.testing.assert_raises(KeyError):
        tree.get_node("root/L/R")


def test_save_load_round_trip_preserves_tree_and_diagnostics(tmp_path) -> None:
    tree, _ = make_test_tree()
    tree.oracle_queries = 123
    tree.metadata["error_metric"] = "relative_l2_error"
    tree.root.metadata["splitter_metadata"] = {
        "soft_loss_history": [1.0, 0.4, 0.2],
        "projected_grad_norm_history": np.array([3.0, 0.8, 0.1]),
    }
    X = np.array([[-1.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
    expected = tree.predict(X)
    path = tmp_path / "nested" / "tree.joblib"

    saved_path = tree.save(
        path,
        run_config={"benchmark": "quadratic", "leaf_degree": 2},
    )
    loaded, run_config = RegressionTree.load(path)

    assert saved_path == path
    assert loaded.oracle_queries == 123
    assert loaded.metadata["error_metric"] == "relative_l2_error"
    assert loaded.num_nodes() == tree.num_nodes()
    assert run_config == {"benchmark": "quadratic", "leaf_degree": 2}
    np.testing.assert_array_equal(loaded.predict(X), expected)
    diagnostics = loaded.root.metadata["splitter_metadata"]
    assert diagnostics["soft_loss_history"] == [1.0, 0.4, 0.2]
    np.testing.assert_array_equal(
        diagnostics["projected_grad_norm_history"],
        np.array([3.0, 0.8, 0.1]),
    )
