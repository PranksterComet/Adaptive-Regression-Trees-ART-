"""Queue scheduling and shared builder policies, independent of CUDA."""

import numpy as np
import pytest

from art.builder import RegressionTreeBuilder
from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.sampling import HitAndRunSampler
from art.splitters import SoftObliqueSplitter
from art.temperature import TemperatureConfig
from art.tree import LeafNode, SplitNode
from test_builder import FixedAxisSplitter


def make_builder(**kwargs):
    options = dict(
        domain=BoxDomain.hypercube(2, -1., 1.),
        oracle=lambda X: X[..., 0]**2 + .2 * X[..., 1]**2,
        oracle_vectorized=True,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=FixedAxisSplitter(median=True),
        error_threshold=0., error_metric=mean_squared_error,
        sample_count=20, max_depth=3,
        sampler=HitAndRunSampler(burn_in=0, thinning=1),
        isotropic_sampling=False, random_state=81, store_samples=True,
    )
    options.update(kwargs)
    return RegressionTreeBuilder(**options)


def observe(builder, monkeypatch):
    prepared, split = [], []
    prepare, dispatch = builder._prepare_node, builder._split_node

    def prepare_node(work):
        prepared.append(work.node_id)
        if work.parent is not None:
            placeholder = getattr(work.parent, work.side)
            assert isinstance(placeholder, LeafNode) and placeholder.status == "pending"
            assert placeholder.node_id == work.node_id
            assert work.samples.n_new == 0
            assert len(work.samples.X) == work.samples.n_inherited
        return prepare(work)

    def split_node(work):
        split.append(work.node_id)
        assert work.model is not None and np.isfinite(work.parent_mse)
        return dispatch(work)

    monkeypatch.setattr(builder, "_prepare_node", prepare_node)
    monkeypatch.setattr(builder, "_split_node", split_node)
    return prepared, split


@pytest.mark.parametrize("traversal", ["dfs", "bfs"])
def test_traversal_order_and_complete_child_attachment(monkeypatch, traversal):
    builder = make_builder(traversal=traversal, node_batch_size=3)
    prepared, split = observe(builder, monkeypatch)
    result = builder.build()
    nodes = list(result.tree.iter_nodes())
    expected = nodes if traversal == "dfs" else sorted(nodes, key=lambda n: (n.depth, n.node_id))
    assert prepared == [n.node_id for n in expected]
    assert split == [n.node_id for n in expected if isinstance(n, SplitNode)]
    assert result.tree.num_nodes() == 15 and result.tree.num_leaves() == 8
    assert result.tree.max_depth() == 3
    assert all(n.status != "pending" for n in nodes)
    assert result.tree.metadata["traversal"] == traversal
    assert result.tree.metadata["node_batch_size"] == (3 if traversal == "bfs" else 1)
    assert np.isfinite(result.tree.predict(np.zeros((3, 2)))).all()


@pytest.mark.parametrize("size,expected", [(1, [1] * 7), (3, [1, 2, 3, 1]), (20, [1, 2, 4])])
def test_bfs_forms_bounded_cohorts_and_defers_child_topups(monkeypatch, size, expected):
    from art import _batched_builder as queue

    builder = make_builder(traversal="bfs", node_batch_size=size)
    groups, dispatch_queries = [], []
    run = queue.build_bfs

    def dispatch(ready):
        groups.append([work.node_id for work in ready])
        dispatch_queries.append(builder.oracle_queries)
        return [builder._split_node(work) for work in ready]

    monkeypatch.setattr(queue, "build_bfs", lambda b, root: run(b, root, dispatch))
    children = builder._children

    def child_jobs(work, node):
        queries = builder.oracle_queries
        jobs = list(children(work, node))
        assert builder.oracle_queries == queries
        assert all(job.samples.n_new == 0 for job in jobs)
        return iter(jobs)

    monkeypatch.setattr(builder, "_children", child_jobs)
    result = builder.build()
    assert [len(group) for group in groups] == expected
    assert all(len(group) <= size for group in groups)
    assert dispatch_queries[0] == builder.target_samples  # No child top-up before root split.
    assert result.oracle_queries == 160


def test_bfs_inherits_samples_and_queries_only_missing_rows():
    calls = []

    def oracle(X):
        calls.append(X.copy())
        return X[:, 0]**2 + .2 * X[:, 1]**2

    result = make_builder(traversal="bfs", oracle=oracle).build()
    assert result.oracle_queries == sum(len(X) for X in calls) == 160
    assert [len(X) for X in calls] == [20] + [10] * 14
    for node in result.tree.iter_nodes():
        X, y = node.metadata["X"], node.metadata["y"]
        assert len(X) == 20 and node.region.contains(X).all()
        if isinstance(node, SplitNode):
            right = X @ node.w >= node.z
            for child, mask in ((node.left, ~right), (node.right, right)):
                inherited = child.metadata["n_inherited"]
                assert inherited == int(mask.sum())
                np.testing.assert_array_equal(child.metadata["X"][:inherited], X[mask])
                np.testing.assert_array_equal(child.metadata["y"][:inherited], y[mask])


def test_terminal_nodes_do_not_occupy_ready_slots(monkeypatch):
    from art import _batched_builder as queue

    class RootAxisThenMedian(FixedAxisSplitter):
        def split(self, X, *args, **kwargs):
            self.median = not (np.min(X[:, 0]) < 0 < np.max(X[:, 0]))
            return super().split(X, *args, **kwargs)

    builder = make_builder(
        traversal="bfs", node_batch_size=2, splitter=RootAxisThenMedian(),
        oracle=lambda X: np.where(X[:, 0] < 0, X[:, 0], X[:, 0]**2),
        error_threshold=1e-20,
    )
    groups, run = [], queue.build_bfs

    def dispatch(ready):
        groups.append([work.node_id for work in ready])
        return [builder._split_node(work) for work in ready]

    monkeypatch.setattr(queue, "build_bfs", lambda b, root: run(b, root, dispatch))
    result = builder.build()
    assert groups == [["root"], ["root/R"], ["root/R/L", "root/R/R"]]
    assert result.tree.get_node("root/L").status == "tolerance_met"
    assert all(n.status != "pending" for n in result.tree.iter_nodes())


@pytest.mark.parametrize("traversal", ["dfs", "bfs"])
def test_sampling_failure_keeps_inherited_fit_without_new_queries(traversal):
    class FailingSampler:
        def sample(self, *args, **kwargs):
            raise RuntimeError("scripted sampling failure")

    result = make_builder(traversal=traversal, sampler=FailingSampler()).build()
    assert isinstance(result.tree.root, SplitNode) and result.oracle_queries == 20
    for leaf in result.tree.iter_leaves():
        assert leaf.status == "sampling_failed"
        assert leaf.metadata["n_new"] == 0 and leaf.metadata["n_inherited"] == 10
        assert "scripted sampling failure" in leaf.metadata["sampling_error"]
        assert np.isfinite(leaf.predict_one(np.mean(leaf.metadata["X"], axis=0)))


@pytest.mark.parametrize("store", [False, True])
def test_bfs_storage_and_serialization(tmp_path, store):
    from art.tree import RegressionTree

    result = make_builder(traversal="bfs", store_samples=store,
                          store_leaf_representative=store).build()
    for node in result.tree.iter_nodes():
        assert ("X" in node.metadata) == store and ("y" in node.metadata) == store
        if isinstance(node, LeafNode):
            assert ("sample_centroid" in node.metadata) == store
    path = tmp_path / "bfs.joblib"
    result.tree.save(path)
    loaded, _ = RegressionTree.load(path)
    X = np.random.default_rng(82).uniform(-1, 1, (31, 2))
    np.testing.assert_array_equal(loaded.predict(X), result.tree.predict(X))
    assert loaded.metadata["traversal"] == "bfs"


def test_dispatch_must_return_one_result_per_ready_node(monkeypatch):
    from art import _batched_builder as queue

    run = queue.build_bfs
    monkeypatch.setattr(queue, "build_bfs", lambda b, root: run(b, root, lambda ready: []))
    with pytest.raises(RuntimeError, match="one node per ready job"):
        make_builder(traversal="bfs").build()


@pytest.mark.parametrize("kwargs", [{"traversal": "invalid"}, {"traversal": "BFS"},
                                    {"node_batch_size": 0}, {"node_batch_size": 1.5},
                                    {"node_batch_size": True}])
def test_invalid_scheduling_options(kwargs):
    with pytest.raises(ValueError):
        make_builder(**kwargs)


def test_default_traversal_remains_dfs(monkeypatch):
    builder = make_builder()
    prepared, _ = observe(builder, monkeypatch)
    result = builder.build()
    assert builder.traversal == "dfs"
    assert prepared == [node.node_id for node in result.tree.iter_nodes()]


@pytest.mark.parametrize("degree", [1, 2])
def test_bfs_with_real_cpu_soft_splitter(monkeypatch, degree):
    model = AffineRidgeModel(ridge=1e-6) if degree == 1 else PolynomialRidgeModel(ridge=1e-6)
    splitter = SoftObliqueSplitter(
        model.clone(), temperature=.2, optimizer="cpu", max_iters=12,
        min_side_points=3, min_side_fallback=True, n_restarts=1,
    )
    builder = make_builder(
        traversal="bfs", node_batch_size=3, model_template=model, splitter=splitter,
        sample_count=40, max_depth=2, max_retries_on_failure=1,
        check_split_gain=True,
        store_diagnostics=True,
        temperature_config=TemperatureConfig(strategy="splitter"),
        oracle=lambda X: np.where(X[:, 0] >= 0,
                                 2. + X[:, 0]**2 + .3 * X[:, 1],
                                 -.5 * X[:, 0] + .2 * X[:, 1]**2),
    )
    prepared, split = observe(builder, monkeypatch)
    result = builder.build()
    nodes = list(result.tree.iter_nodes())
    assert isinstance(result.tree.root, SplitNode) and result.tree.num_split_nodes() >= 1
    assert result.tree.root.metadata["splitter_metadata"]["n_iters"] > 0
    assert split and all(node.status != "pending" for node in nodes)
    assert prepared == [node.node_id for node in sorted(nodes, key=lambda n: (n.depth, n.node_id))]
    assert result.oracle_queries == sum(node.metadata["n_new"] for node in nodes)
    for node in nodes:
        assert node.region.contains(node.metadata["X"]).all()
        if isinstance(node, SplitNode):
            assert node.metadata["split_gain_mse"] > 0
    X = np.random.default_rng(83).uniform(-1, 1, (101, 2))
    assert np.isfinite(result.tree.predict(X)).all()
