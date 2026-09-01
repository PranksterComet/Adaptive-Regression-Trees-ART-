from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from art.builder import RegressionTreeBuilder
from art.domain import BoxDomain
from art.metrics import mean_squared_error
from art.models import (
    AffineRidgeModel,
    PolynomialRidgeModel,
    PreparedFeatureModel,
)
from art.sampling import HitAndRunSampler
from art.splitters import SplitNotFoundError, SplitResult
from art.temperature import DEFAULT_TEMPERATURE_GRID, TemperatureConfig
from art.tree import LeafNode, SplitNode


@dataclass
class FixedAxisSplitter:
    """Deterministic splitter used to isolate builder behavior."""

    ridge: float = 1e-12
    converged: bool = True
    n_iters: int = 1
    stop_reason: str = "converged"
    metadata: dict[str, object] = field(default_factory=dict)
    random_state: int | np.random.Generator | None = None

    def split(
        self,
        X,
        y,
        parent_model=None,
        parent_loss=None,
        prepared_design=None,
    ) -> SplitResult:
        w = np.zeros(X.shape[1])
        w[0] = 1.0
        z = 0.0
        right = (X @ w - z) >= 0.0
        if prepared_design is None:
            left_model = AffineRidgeModel(self.ridge).fit(X[~right], y[~right])
            right_model = AffineRidgeModel(self.ridge).fit(X[right], y[right])
            left_predictions = left_model.predict(X[~right])
            right_predictions = right_model.predict(X[right])
        else:
            left_model = parent_model.clone()
            right_model = parent_model.clone()
            assert isinstance(left_model, PreparedFeatureModel)
            assert isinstance(right_model, PreparedFeatureModel)
            left_design = prepared_design.subset(~right)
            right_design = prepared_design.subset(right)
            left_model.fit_design(left_design, y[~right])
            right_model.fit_design(right_design, y[right])
            left_predictions = left_model.predict_design(left_design)
            right_predictions = right_model.predict_design(right_design)
        loss = (
            np.sum(~right)
            / X.shape[0]
            * mean_squared_error(y[~right], left_predictions)
            + np.sum(right)
            / X.shape[0]
            * mean_squared_error(y[right], right_predictions)
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
            converged=self.converged,
            n_iters=self.n_iters,
            stop_reason=self.stop_reason,
            metadata=dict(self.metadata),
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

    def split(
        self,
        X,
        y,
        parent_model=None,
        parent_loss=None,
        prepared_design=None,
    ) -> SplitResult:
        call = int(self.state["calls"]) + 1
        self.state["calls"] = call
        self.state["X"].append(np.asarray(X).copy())
        self.state["y"].append(np.asarray(y).copy())
        self.state["seeds"].append(self.random_state)
        self.state.setdefault("prepared_design_ids", []).append(
            None if prepared_design is None else id(prepared_design)
        )
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

        result = FixedAxisSplitter().split(
            X,
            y,
            parent_model,
            parent_loss,
            prepared_design,
        )
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


class CountingPolynomialModel(PolynomialRidgeModel):
    """Polynomial model exposing accidental raw feature transformations."""

    def __init__(self, counter: dict[str, int], **kwargs):
        self.counter = counter
        super().__init__(**kwargs)

    def prepare_design(self, X):
        self.counter["prepare_design"] = self.counter.get("prepare_design", 0) + 1
        return super().prepare_design(X)

    def fit(self, X, y):
        self.counter["raw_fit"] = self.counter.get("raw_fit", 0) + 1
        return super().fit(X, y)

    def fit_weighted(self, X, y, weights, weight_floor=1e-12):
        self.counter["raw_weighted_fit"] = (
            self.counter.get("raw_weighted_fit", 0) + 1
        )
        return super().fit_weighted(X, y, weights, weight_floor)

    def clone(self):
        return CountingPolynomialModel(
            self.counter,
            degree=self.degree,
            ridge=self.ridge,
            include_bias=self.include_bias,
            solver=self.solver,
            auto_rcond_threshold=self.auto_rcond_threshold,
        )

    def __deepcopy__(self, memo):
        return self.clone()


def piecewise_oracle(x: np.ndarray) -> float:
    if x[0] < 0.0:
        return x[0] - 0.5 * x[1]
    return 3.0 - 0.25 * x[0] + x[1]


def make_retry_builder(splitter, **kwargs) -> RegressionTreeBuilder:
    model_template = kwargs.pop(
        "model_template",
        AffineRidgeModel(ridge=1e-12),
    )
    return RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        oracle=piecewise_oracle,
        model_template=model_template,
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
    assert DEFAULT_TEMPERATURE_GRID == (
        1e-4,
        5e-4,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
    )


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
    assert result.build_timing is None
    assert result.tree.metadata["build_timing"] is None


def test_optional_build_timing_profiles_sampling_and_splitter() -> None:
    result = make_retry_builder(
        FixedAxisSplitter(),
        profile_build_timing=True,
    ).build()
    timing = result.build_timing

    assert timing is not None
    assert result.tree.metadata["build_timing"] == timing
    assert timing["total_build_seconds"] > 0.0
    assert timing["sampling_seconds"] > 0.0
    assert timing["splitter_seconds"] > 0.0
    assert timing["optimizer_model_refit_seconds"] == 0.0
    assert timing["sampling_percent"] > 0.0
    assert timing["splitter_percent"] > 0.0
    assert np.isclose(
        timing["sampling_percent"]
        + timing["splitter_percent"]
        + timing["other_percent"],
        100.0,
    )


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
    assert root.metadata["sampling_method"] == "exact_uniform_box"
    for child in (root.left, root.right):
        assert child.metadata["sampling_method"] == "isotropic_hit_and_run"
        assert child.metadata["sampling_pilot_length"] == 100
        assert child.metadata["sampling_thinning"] == 1
        assert np.isfinite(child.metadata["sampling_covariance_condition_number"])

    X_test = np.array([[-0.8, 0.2], [0.6, -0.4]])
    y_test = np.array([oracle(x) for x in X_test])
    assert mean_squared_error(y_test, result.tree.predict(X_test)) < 1e-12


def test_initial_gradient_tolerance_warns_on_accepted_split() -> None:
    root = make_retry_builder(
        FixedAxisSplitter(n_iters=0, stop_reason="gradient_tolerance")
    ).build().tree.root

    assert isinstance(root, SplitNode)
    assert root.status == "split_with_warnings"
    assert root.metadata["split_iterations"] == 0
    assert root.metadata["split_stop_reason"] == "gradient_tolerance"
    assert root.metadata["warnings"] == ("small_init_grad",)


def test_high_model_conditioning_warns_on_accepted_split() -> None:
    root = make_retry_builder(
        FixedAxisSplitter(metadata={"high_model_conditioning": True})
    ).build().tree.root

    assert isinstance(root, SplitNode)
    assert root.status == "split_with_warnings"
    assert root.metadata["warnings"] == ("high_model_conditioning",)


def test_isotropic_sampling_can_be_disabled() -> None:
    builder = RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        oracle=piecewise_oracle,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=FixedAxisSplitter(),
        error_threshold=0.0,
        sample_count=30,
        max_depth=1,
        sampler=HitAndRunSampler(burn_in=0, thinning=3),
        isotropic_sampling=False,
        random_state=3,
    )

    root = builder.build().tree.root

    assert isinstance(root, SplitNode)
    for child in (root.left, root.right):
        assert child.metadata["sampling_method"] == "HitAndRunSampler"
        assert child.metadata["sampling_pilot_length"] == 0
        assert child.metadata["sampling_thinning"] == 3
        assert child.metadata["sampling_covariance_condition_number"] is None


def test_isotropic_sampling_reports_covariance_floor_saturation() -> None:
    builder = RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        oracle=piecewise_oracle,
        model_template=AffineRidgeModel(ridge=1e-12),
        splitter=FixedAxisSplitter(),
        error_threshold=0.0,
        sample_count=30,
        max_depth=1,
        sampler=HitAndRunSampler(
            burn_in=0,
            thinning=1,
            direction_eigenvalue_floor=1.0,
        ),
        random_state=3,
    )

    root = builder.build().tree.root

    assert isinstance(root, SplitNode)
    for child in (root.left, root.right):
        assert child.metadata["sampling_covariance_floor_saturated"] is True
        assert child.metadata["sampling_warnings"] == (
            "covariance_eigenvalue_floor_saturated",
        )


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


def test_polynomial_design_is_reused_across_builder_retries() -> None:
    counter: dict[str, int] = {}
    model = CountingPolynomialModel(counter, degree=2, ridge=1e-10)
    splitter = RetryScriptSplitter(("insufficient_split_gain", "success"))

    result = make_retry_builder(
        splitter,
        model_template=model,
        min_split_gain=1e-8,
        max_retries_on_failure=1,
    ).build()

    assert isinstance(result.tree.root, SplitNode)
    assert counter["prepare_design"] == result.tree.num_nodes() == 3
    assert counter.get("raw_fit", 0) == 0
    assert counter.get("raw_weighted_fit", 0) == 0
    assert splitter.state["prepared_design_ids"][0] == (
        splitter.state["prepared_design_ids"][1]
    )


def test_temperature_tuning_subsets_one_node_design() -> None:
    counter: dict[str, int] = {}
    model = CountingPolynomialModel(counter, degree=2, ridge=1e-10)
    splitter = TemperatureRetryScriptSplitter(("success",))

    result = make_retry_builder(
        splitter,
        model_template=model,
        temperature_config=TemperatureConfig(
            strategy="tune_root",
            c_values=(0.05, 0.1),
            max_points=None,
        ),
    ).build()

    assert isinstance(result.tree.root, SplitNode)
    assert counter["prepare_design"] == result.tree.num_nodes() == 3
    assert counter.get("raw_fit", 0) == 0
    assert counter.get("raw_weighted_fit", 0) == 0
    design_ids = splitter.state["prepared_design_ids"]
    assert len(design_ids) == 3
    assert design_ids[0] == design_ids[1]
    assert design_ids[2] != design_ids[0]
