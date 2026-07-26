from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from art.builder import RegressionTreeBuilder
from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.models import AffineRidgeModel, PolynomialRidgeModel
from art.sampling import HitAndRunSampler
from art.splitters import SplitNotFoundError, SplitResult
from art.temperature import DEFAULT_TEMPERATURE_GRID, TemperatureConfig
from art.tree import LeafNode, SplitNode


@dataclass
class FixedAxisSplitter:
    """Deterministic splitter used to isolate builder behavior."""

    ridge: float = 1e-12
    random_state: int | np.random.Generator | None = None

    def split(self, X, y, parent_model=None, parent_loss=None) -> SplitResult:
        w = np.zeros(X.shape[1])
        w[0] = 1.0
        z = 0.0
        right = (X @ w - z) >= 0.0
        left_model = AffineRidgeModel(self.ridge).fit(X[~right], y[~right])
        right_model = AffineRidgeModel(self.ridge).fit(X[right], y[right])
        loss = (
            np.sum(~right)
            / X.shape[0]
            * mean_squared_error(y[~right], left_model.predict(X[~right]))
            + np.sum(right)
            / X.shape[0]
            * mean_squared_error(y[right], right_model.predict(X[right]))
        )
        gain = float(parent_loss - loss)
        return SplitResult(
            w=w,
            z=z,
            left_model=left_model,
            right_model=right_model,
            loss=loss,
            parent_loss=float(parent_loss),
            split_gain=gain,
            relative_split_gain=gain / max(float(parent_loss), 1e-12),
            n_left=int(np.sum(~right)),
            n_right=int(np.sum(right)),
            converged=True,
            n_iters=1,
            stop_reason="converged",
        )


class RetryScriptSplitter:
    """Splitter with shared scripted outcomes across builder deep copies."""

    def __init__(
        self,
        outcomes: tuple[str, ...],
        state: dict[str, object] | None = None,
        n_restarts: int = 1,
    ):
        self.outcomes = outcomes
        self.state = (
            {"calls": 0, "X": [], "y": [], "seeds": []}
            if state is None
            else state
        )
        self.n_restarts = n_restarts
        self.random_state = None

    def __deepcopy__(self, memo):
        copied = type(self)(self.outcomes, self.state, self.n_restarts)
        copied.random_state = self.random_state
        if hasattr(self, "temperature"):
            copied.temperature = self.temperature
        return copied

    def split(self, X, y, parent_model=None, parent_loss=None) -> SplitResult:
        call = int(self.state["calls"]) + 1
        self.state["calls"] = call
        self.state["X"].append(np.asarray(X).copy())
        self.state["y"].append(np.asarray(y).copy())
        self.state["seeds"].append(self.random_state)
        outcome = self.outcomes[min(call - 1, len(self.outcomes) - 1)]
        diagnostics = {
            "run": call,
            "soft_loss_history": [float(call)],
            "projected_grad_norm_history": [float(call)],
        }

        if outcome == "min_side_points":
            raise SplitNotFoundError(
                "min_side_points",
                "scripted minimum-side failure",
                diagnostics=diagnostics,
            )

        result = FixedAxisSplitter().split(X, y, parent_model, parent_loss)
        result.metadata = diagnostics
        if outcome == "insufficient_split_gain":
            result.loss = float(parent_loss)
            result.split_gain = 0.0
            result.relative_split_gain = 0.0
        return result


class TemperatureRetryScriptSplitter(RetryScriptSplitter):
    def __init__(
        self,
        outcomes: tuple[str, ...],
        state: dict[str, object] | None = None,
        n_restarts: int = 1,
    ):
        super().__init__(outcomes, state, n_restarts)
        self.temperature = 0.1


def piecewise_oracle(x: np.ndarray) -> float:
    if x[0] < 0.0:
        return x[0] - 0.5 * x[1]
    return 3.0 - 0.25 * x[0] + x[1]


def make_retry_builder(splitter, **kwargs) -> RegressionTreeBuilder:
    return RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        oracle=piecewise_oracle,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=splitter,
        error_threshold=0.0,
        sample_count=30,
        max_depth=1,
        sampler=HitAndRunSampler(burn_in=0, thinning=1),
        store_diagnostics=True,
        random_state=3,
        **kwargs,
    )


def test_model_effective_dimensions_and_temperature_grid() -> None:
    assert AffineRidgeModel().effective_dimension(3) == 4
    assert PolynomialRidgeModel(degree=2).effective_dimension(3) == 10
    assert PolynomialRidgeModel(degree=2, include_bias=False).effective_dimension(3) == 9
    assert DEFAULT_TEMPERATURE_GRID == (1e-4, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)


def test_root_stops_and_counts_only_sampled_oracle_points() -> None:
    domain = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    builder = RegressionTreeBuilder(
        domain=domain,
        oracle=lambda x: 1.5 * x[0] - 0.25 * x[1] + 0.75,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=FixedAxisSplitter(),
        error_threshold=1e-8,
        sample_count=20,
        max_depth=3,
        random_state=2,
    )

    result = builder.build()

    assert isinstance(result.tree.root, LeafNode)
    assert result.tree.root.status == "tolerance_met"
    assert result.oracle_queries == 20
    assert result.tree.oracle_queries == 20
    assert "X" not in result.tree.root.metadata
    assert "y" not in result.tree.root.metadata


def test_children_inherit_samples_and_query_only_the_top_up() -> None:
    domain = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]]))

    def oracle(x: np.ndarray) -> float:
        if x[0] < 0.0:
            return x[0] - 0.5 * x[1]
        return 3.0 - 0.25 * x[0] + x[1]

    builder = RegressionTreeBuilder(
        domain=domain,
        oracle=oracle,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=FixedAxisSplitter(),
        error_threshold=1e-8,
        sample_count=30,
        max_depth=1,
        sampler=HitAndRunSampler(burn_in=0, thinning=1),
        random_state=3,
    )

    result = builder.build()
    root = result.tree.root

    assert isinstance(root, SplitNode)
    assert root.metadata["split_gain_mse"] > 0.0
    assert root.metadata["relative_split_gain_mse"] > 0.0
    assert result.oracle_queries == 60
    assert [node.node_id for node in result.tree.iter_nodes()] == ["root", "root/L", "root/R"]
    assert [leaf.node_id for leaf in result.tree.iter_leaves()] == ["root/L", "root/R"]
    assert result.tree.num_nodes() == 3
    assert result.tree.num_split_nodes() == 1
    assert result.tree.num_leaves() == 2
    assert root.left.metadata["n_samples"] == 30
    assert root.right.metadata["n_samples"] == 30
    assert root.left.metadata["n_new"] + root.right.metadata["n_new"] == 30

    X_test = np.array([[-0.8, 0.2], [0.6, -0.4]])
    y_test = np.array([oracle(x) for x in X_test])
    assert mean_squared_error(y_test, result.tree.predict(X_test)) < 1e-12


def test_store_samples_toggle_keeps_copies_on_nodes() -> None:
    domain = BoxDomain(np.array([[-1.0, 1.0]]))
    builder = RegressionTreeBuilder(
        domain=domain,
        oracle=lambda x: x[0] ** 2,
        model_template=AffineRidgeModel(),
        splitter=FixedAxisSplitter(),
        error_threshold=0.0,
        sample_count=12,
        max_depth=0,
        store_samples=True,
        random_state=4,
    )

    root = builder.build().tree.root

    assert isinstance(root, LeafNode)
    assert root.status == "max_depth"
    assert root.metadata["X"].shape == (12, 1)
    assert root.metadata["y"].shape == (12,)


def test_min_side_failure_retries_with_same_samples_and_new_seed() -> None:
    splitter = RetryScriptSplitter(("min_side_points", "success"))
    result = make_retry_builder(
        splitter,
        max_retries_on_failure=2,
    ).build()
    root = result.tree.root

    assert isinstance(root, SplitNode)
    assert result.restarts_on_failure == 1
    assert result.tree.metadata["restarts_on_failure"] == 1
    assert root.metadata["restarts_on_failure"] == 1
    assert root.metadata["split_attempt_failure_reasons"] == ("min_side_points",)
    assert root.metadata["splitter_metadata"]["run"] == 2
    assert np.array_equal(splitter.state["X"][0], splitter.state["X"][1])
    assert np.array_equal(splitter.state["y"][0], splitter.state["y"][1])
    assert splitter.state["seeds"][0] != splitter.state["seeds"][1]


def test_insufficient_gain_retries_and_accepts_later_split() -> None:
    splitter = RetryScriptSplitter(("insufficient_split_gain", "success"))
    result = make_retry_builder(
        splitter,
        min_split_gain=1e-6,
        max_retries_on_failure=1,
    ).build()
    root = result.tree.root

    assert isinstance(root, SplitNode)
    assert result.restarts_on_failure == 1
    assert root.metadata["split_attempt_failure_reasons"] == (
        "insufficient_split_gain",
    )
    assert root.metadata["splitter_metadata"]["run"] == 2


def test_exhausted_failure_stores_last_run_diagnostics() -> None:
    splitter = RetryScriptSplitter(("min_side_points",))
    result = make_retry_builder(
        splitter,
        max_retries_on_failure=2,
    ).build()
    root = result.tree.root

    assert isinstance(root, LeafNode)
    assert root.status == "min_side_points"
    assert result.restarts_on_failure == 2
    assert result.oracle_queries == 30
    assert root.metadata["restarts_on_failure"] == 2
    assert root.metadata["split_attempt_failure_reasons"] == (
        "min_side_points",
        "min_side_points",
        "min_side_points",
    )
    assert root.metadata["splitter_metadata"]["run"] == 3


def test_exhausted_gain_retries_store_last_result_diagnostics() -> None:
    splitter = RetryScriptSplitter(("insufficient_split_gain",))
    result = make_retry_builder(
        splitter,
        min_split_gain=1e-6,
        max_retries_on_failure=1,
    ).build()
    root = result.tree.root

    assert isinstance(root, LeafNode)
    assert root.status == "insufficient_split_gain"
    assert result.restarts_on_failure == 1
    assert root.metadata["restarts_on_failure"] == 1
    assert root.metadata["splitter_metadata"]["run"] == 2


def test_failure_retries_require_single_splitter_restart() -> None:
    splitter = RetryScriptSplitter(("success",), n_restarts=2)

    with np.testing.assert_raises_regex(
        ValueError,
        "requires splitter.n_restarts == 1",
    ):
        make_retry_builder(splitter, max_retries_on_failure=1)


def test_active_temperature_tuning_disables_failure_retries() -> None:
    splitter = TemperatureRetryScriptSplitter(("success", "min_side_points"))
    result = make_retry_builder(
        splitter,
        max_retries_on_failure=3,
        temperature_config=TemperatureConfig(
            strategy="tune_root",
            c_values=(0.1,),
            max_points=None,
        ),
    ).build()
    root = result.tree.root

    assert isinstance(root, LeafNode)
    assert root.status == "min_side_points"
    assert result.restarts_on_failure == 0
    assert splitter.state["calls"] == 2
    assert root.metadata["temperature_tuned_at_node"] is True
    assert root.metadata["splitter_metadata"]["run"] == 2
