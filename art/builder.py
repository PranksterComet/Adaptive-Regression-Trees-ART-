"""Recursive construction of adaptive regression trees."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .domain import BoxDomain, PolytopeRegion, split_region
from .metrics import mean_squared_error, relative_l2_error
from .models import (
    AffineRidgeModel,
    RegressionModel,
    WeightedRegressionModel,
    model_effective_dimension,
)
from .sampling import HitAndRunSampler, Sampler, sample_uniform_box
from .splitters import (
    HingeAffineSplitter,
    SoftObliqueSplitter,
    SplitNotFoundError,
    SplitResult,
    Splitter,
)
from .temperature import TemperatureConfig, estimate_temperature
from .tree import LeafNode, RegressionTree, SplitNode, TreeNode


ErrorMetric = Callable[[np.ndarray, np.ndarray], float]
SampleCountPolicy = Callable[[RegressionModel, int], int]


@dataclass(frozen=True)
class TreeBuildResult:
    tree: RegressionTree
    oracle_queries: int
    restarts_on_failure: int


@dataclass
class _NodeSamples:
    X: np.ndarray
    y: np.ndarray
    n_inherited: int
    n_new: int


@dataclass
class RegressionTreeBuilder:
    """Build a regression tree from an oracle over a convex domain."""

    domain: BoxDomain | PolytopeRegion
    oracle: Callable[[np.ndarray], float | np.ndarray]
    model_template: RegressionModel
    splitter: Splitter
    error_threshold: float
    max_depth: int = 10
    error_metric: ErrorMetric = relative_l2_error
    sample_multiplier: int = 50
    sample_count: int | SampleCountPolicy | None = None
    sampler: Sampler = field(
        default_factory=lambda: HitAndRunSampler(burn_in=0, thinning=20)
    )
    root_sampler: Sampler = field(default_factory=HitAndRunSampler)
    exact_box_root: bool = True
    temperature_config: TemperatureConfig | None = None
    min_split_gain: float = 0.0
    min_relative_split_gain: float = 0.0
    max_retries_on_failure: int = 0
    store_samples: bool = False
    store_diagnostics: bool = False
    oracle_vectorized: bool = False
    random_state: int | np.random.Generator | None = None

    oracle_queries: int = field(default=0, init=False)
    restarts_on_failure: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _root_temperature_c: float | None = field(default=None, init=False, repr=False)
    _target_samples: int = field(init=False, repr=False)
    _effective_dimension: int | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.error_threshold < 0.0:
            raise ValueError("error_threshold must be nonnegative.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.sample_multiplier < 1:
            raise ValueError("sample_multiplier must be at least 1.")
        if self.min_split_gain < 0.0 or self.min_relative_split_gain < 0.0:
            raise ValueError("Split-gain thresholds must be nonnegative.")
        if self.max_retries_on_failure < 0:
            raise ValueError("max_retries_on_failure must be nonnegative.")
        if (
            self.max_retries_on_failure > 0
            and int(getattr(self.splitter, "n_restarts", 1)) != 1
        ):
            raise ValueError(
                "max_retries_on_failure requires splitter.n_restarts == 1."
            )

        self._rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        self._effective_dimension = self._resolve_effective_dimension()
        self._target_samples = self._resolve_sample_count()
        if self._target_samples < 2:
            raise ValueError("The node sample count must be at least 2.")

        splitter_uses_temperature = hasattr(self.splitter, "temperature")
        if isinstance(self.splitter, HingeAffineSplitter) and not isinstance(
            self.model_template, AffineRidgeModel
        ):
            raise TypeError("HingeAffineSplitter requires an AffineRidgeModel.")
        if isinstance(self.splitter, SoftObliqueSplitter) and not isinstance(
            self.model_template, WeightedRegressionModel
        ):
            raise TypeError(
                "SoftObliqueSplitter requires a model with fit_weighted support."
            )
        if splitter_uses_temperature and self.temperature_config is None:
            self.temperature_config = TemperatureConfig()
        if not splitter_uses_temperature and self.temperature_config is not None:
            raise ValueError("temperature_config requires a splitter with a temperature attribute.")
        if (
            self.temperature_config is not None
            and self.temperature_config.strategy in ("tune_root", "tune_node")
            and self._target_samples < 4
        ):
            raise ValueError("Temperature tuning requires at least four node samples.")

    @property
    def dimension(self) -> int:
        return self.domain.dimension

    @property
    def target_samples(self) -> int:
        return self._target_samples

    def build(self) -> TreeBuildResult:
        """Build a fresh tree and reset all per-build state."""

        self.oracle_queries = 0
        self.restarts_on_failure = 0
        self._root_temperature_c = None
        root_region = self.domain.as_region() if isinstance(self.domain, BoxDomain) else self.domain
        X_root = self._sample_root(root_region)
        y_root = self._query_oracle(X_root)
        root_samples = _NodeSamples(
            X=X_root,
            y=y_root,
            n_inherited=0,
            n_new=X_root.shape[0],
        )
        root = self._build_node(root_region, root_samples, depth=0, node_id="root")
        tree = RegressionTree(
            root=root,
            oracle_queries=self.oracle_queries,
            metadata={
                "error_metric": self._metric_name(),
                "error_threshold": self.error_threshold,
                "effective_dimension": self._effective_dimension,
                "target_samples_per_node": self._target_samples,
                "restarts_on_failure": self.restarts_on_failure,
                "temperature_strategy": (
                    None if self.temperature_config is None else self.temperature_config.strategy
                ),
            },
        )
        return TreeBuildResult(
            tree=tree,
            oracle_queries=self.oracle_queries,
            restarts_on_failure=self.restarts_on_failure,
        )

    def _build_node(
        self,
        region: PolytopeRegion,
        samples: _NodeSamples,
        depth: int,
        node_id: str,
    ) -> TreeNode:
        model = self.model_template.clone().fit(samples.X, samples.y)
        predictions = model.predict(samples.X)
        fit_error = float(self.error_metric(samples.y, predictions))
        parent_mse = mean_squared_error(samples.y, predictions)
        base_metadata = self._node_metadata(samples, fit_error, parent_mse)

        if fit_error <= self.error_threshold:
            return self._make_leaf(
                model, region, depth, node_id, "tolerance_met", base_metadata, samples
            )
        if depth >= self.max_depth:
            return self._make_leaf(
                model, region, depth, node_id, "max_depth", base_metadata, samples
            )

        try:
            split_result, temperature_metadata, retry_metadata = self._fit_split(
                samples.X,
                samples.y,
                model,
                parent_mse,
                depth,
            )
        except SplitNotFoundError as exc:
            metadata = {
                **base_metadata,
                **exc.context,
                "split_failure_reason": exc.reason,
                "restarts_on_failure": exc.restarts_on_failure,
                "split_attempt_failure_reasons": exc.failure_reasons,
            }
            if self.store_diagnostics and exc.diagnostics is not None:
                metadata["splitter_metadata"] = copy.deepcopy(exc.diagnostics)
            return self._make_leaf(
                model, region, depth, node_id, exc.reason, metadata, samples
            )

        rejection_reason = self._split_rejection_reason(split_result)
        if rejection_reason is not None:
            metadata = {
                **base_metadata,
                **self._split_summary(split_result),
                **temperature_metadata,
                **retry_metadata,
            }
            if self.store_diagnostics:
                metadata["splitter_metadata"] = copy.deepcopy(split_result.metadata)
            return self._make_leaf(
                model, region, depth, node_id, rejection_reason, metadata, samples
            )

        right_mask = (samples.X @ split_result.w - split_result.z) >= 0.0
        left_region, right_region = split_region(
            region,
            split_result.w,
            split_result.z,
            depth=depth + 1,
            left_tag=f"{node_id}/L",
            right_tag=f"{node_id}/R",
        )
        left = self._build_child(
            left_region,
            samples.X[~right_mask],
            samples.y[~right_mask],
            depth + 1,
            f"{node_id}/L",
        )
        right = self._build_child(
            right_region,
            samples.X[right_mask],
            samples.y[right_mask],
            depth + 1,
            f"{node_id}/R",
        )

        warnings = self._split_warnings(split_result)
        metadata = {
            **base_metadata,
            **self._split_summary(split_result),
            **temperature_metadata,
            **retry_metadata,
            "warnings": warnings,
        }
        if self.store_diagnostics:
            metadata["splitter_metadata"] = copy.deepcopy(split_result.metadata)
        self._store_samples(metadata, samples)
        return SplitNode(
            w=split_result.w,
            z=split_result.z,
            left=left,
            right=right,
            region=region,
            depth=depth,
            node_id=node_id,
            status="split_with_warnings" if warnings else "split",
            metadata=metadata,
        )

    def _build_child(
        self,
        region: PolytopeRegion,
        inherited_X: np.ndarray,
        inherited_y: np.ndarray,
        depth: int,
        node_id: str,
    ) -> TreeNode:
        try:
            samples = self._top_up_child(region, inherited_X, inherited_y)
        except (RuntimeError, ValueError) as exc:
            model = self.model_template.clone().fit(inherited_X, inherited_y)
            predictions = model.predict(inherited_X)
            fallback = _NodeSamples(
                X=inherited_X,
                y=inherited_y,
                n_inherited=inherited_X.shape[0],
                n_new=0,
            )
            metadata = self._node_metadata(
                fallback,
                float(self.error_metric(inherited_y, predictions)),
                mean_squared_error(inherited_y, predictions),
            )
            metadata["sampling_error"] = repr(exc)
            return self._make_leaf(
                model, region, depth, node_id, "sampling_failed", metadata, fallback
            )
        return self._build_node(region, samples, depth, node_id)

    def _top_up_child(
        self,
        region: PolytopeRegion,
        inherited_X: np.ndarray,
        inherited_y: np.ndarray,
    ) -> _NodeSamples:
        inherited_X = np.asarray(inherited_X, dtype=float)
        inherited_y = np.asarray(inherited_y, dtype=float).reshape(-1)
        if inherited_X.shape[0] == 0:
            raise RuntimeError("Cannot seed child sampling without an inherited point.")

        if inherited_X.shape[0] > self._target_samples:
            keep = self._rng.choice(
                inherited_X.shape[0], size=self._target_samples, replace=False
            )
            inherited_X = inherited_X[keep]
            inherited_y = inherited_y[keep]

        n_inherited = inherited_X.shape[0]
        n_new = self._target_samples - n_inherited
        if n_new == 0:
            return _NodeSamples(inherited_X, inherited_y, n_inherited, 0)

        x0 = inherited_X[int(self._rng.integers(0, n_inherited))]
        X_new = np.asarray(
            self.sampler.sample(
                region,
                n_new,
                random_state=self._next_seed(),
                x0=x0,
            ),
            dtype=float,
        )
        if X_new.shape != (n_new, self.dimension):
            raise RuntimeError(
                f"Sampler returned shape {X_new.shape}; expected {(n_new, self.dimension)}."
            )
        if not np.all(region.contains(X_new)):
            raise RuntimeError("Sampler returned points outside the child region.")
        y_new = self._query_oracle(X_new)
        return _NodeSamples(
            X=np.vstack([inherited_X, X_new]),
            y=np.concatenate([inherited_y, y_new]),
            n_inherited=n_inherited,
            n_new=n_new,
        )

    def _fit_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel,
        parent_mse: float,
        depth: int,
    ) -> tuple[SplitResult, dict[str, object], dict[str, object]]:
        temperature, temperature_metadata, temperature_tuned = (
            self._resolve_split_temperature(X, y, depth)
        )
        allowed_retries = 0 if temperature_tuned else self.max_retries_on_failure
        failure_reasons = []
        retries = 0

        while True:
            try:
                result = self._run_splitter(
                    X,
                    y,
                    parent_model,
                    parent_mse,
                    temperature,
                )
            except SplitNotFoundError as exc:
                failure_reasons.append(exc.reason)
                if exc.reason == "min_side_points" and retries < allowed_retries:
                    retries += 1
                    self.restarts_on_failure += 1
                    continue
                raise SplitNotFoundError(
                    exc.reason,
                    str(exc),
                    diagnostics=exc.diagnostics,
                    restarts_on_failure=retries,
                    failure_reasons=tuple(failure_reasons),
                    context=temperature_metadata,
                ) from exc

            rejection_reason = self._split_rejection_reason(result)
            if rejection_reason is not None:
                failure_reasons.append(rejection_reason)
            if (
                rejection_reason
                in ("insufficient_split_gain", "insufficient_relative_split_gain")
                and retries < allowed_retries
            ):
                retries += 1
                self.restarts_on_failure += 1
                continue

            retry_metadata = {
                "restarts_on_failure": retries,
                "split_attempt_failure_reasons": tuple(failure_reasons),
            }
            return result, temperature_metadata, retry_metadata

    def _resolve_split_temperature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> tuple[float | None, dict[str, object], bool]:
        config = self.temperature_config
        if config is None or config.strategy == "splitter":
            temperature = getattr(self.splitter, "temperature", None)
            return (
                None,
                {
                    "temperature": temperature,
                    "temperature_c": None,
                    "temperature_tuned_at_node": False,
                },
                False,
            )

        candidate_scores = None
        if config.strategy == "fixed":
            c = config.c
            temperature_tuned = False
        elif config.strategy == "tune_root" and self._root_temperature_c is not None:
            c = self._root_temperature_c
            temperature_tuned = False
        else:
            c, candidate_scores = self._tune_temperature(X, y)
            temperature_tuned = True
            if config.strategy == "tune_root" and depth == 0:
                self._root_temperature_c = c

        temperature = self._temperature_for_c(X, c)
        metadata: dict[str, object] = {
            "temperature": temperature,
            "temperature_c": c,
            "temperature_scale_mode": config.scale_mode,
            "temperature_tuned_at_node": temperature_tuned,
        }
        if self.store_diagnostics and temperature_tuned:
            metadata["temperature_candidate_scores"] = candidate_scores
        return temperature, metadata, temperature_tuned

    def _tune_temperature(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, list[dict[str, object]]]:
        config = self.temperature_config
        n_val = max(1, int(round(config.validation_fraction * X.shape[0])))
        permutation = self._rng.permutation(X.shape[0])
        val_idx = permutation[:n_val]
        fit_idx = permutation[n_val:]
        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        parent_model = self.model_template.clone().fit(X_fit, y_fit)
        parent_mse = mean_squared_error(y_fit, parent_model.predict(X_fit))
        comparison_seed = self._next_seed()
        temperature_seed = self._next_seed()
        candidates = []
        best: tuple[float, float] | None = None

        for c in config.c_values:
            temperature = self._temperature_for_c(
                X_fit, c, random_seed=temperature_seed
            )
            try:
                result = self._run_splitter(
                    X_fit,
                    y_fit,
                    parent_model,
                    parent_mse,
                    temperature,
                    random_seed=comparison_seed,
                )
            except SplitNotFoundError as exc:
                candidates.append(
                    {
                        "c": c,
                        "temperature": temperature,
                        "validation_mse": np.inf,
                        "failure_reason": exc.reason,
                    }
                )
                continue

            rejection_reason = self._split_rejection_reason(result)
            if rejection_reason is not None:
                candidates.append(
                    {
                        "c": c,
                        "temperature": temperature,
                        "validation_mse": np.inf,
                        "failure_reason": rejection_reason,
                    }
                )
                continue

            validation_mse = mean_squared_error(
                y_val, result.predict(X_val)
            )
            candidates.append(
                {
                    "c": c,
                    "temperature": temperature,
                    "validation_mse": validation_mse,
                    "failure_reason": None,
                }
            )
            if best is None or validation_mse < best[0]:
                best = (validation_mse, float(c))

        if best is None:
            raise SplitNotFoundError(
                "temperature_tuning_failed",
                "All temperature candidates failed to produce a valid split.",
            )
        return best[1], candidates

    def _run_splitter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel,
        parent_mse: float,
        temperature: float | None,
        random_seed: int | None = None,
    ) -> SplitResult:
        node_splitter = copy.deepcopy(self.splitter)
        if temperature is not None:
            setattr(node_splitter, "temperature", float(temperature))
        if hasattr(node_splitter, "random_state"):
            setattr(
                node_splitter,
                "random_state",
                self._next_seed() if random_seed is None else random_seed,
            )
        return node_splitter.split(
            X,
            y,
            parent_model=parent_model,
            parent_loss=parent_mse,
        )

    def _temperature_for_c(
        self,
        X: np.ndarray,
        c: float,
        random_seed: int | None = None,
    ) -> float:
        config = self.temperature_config
        nn_method = config.nn_method
        if nn_method is None:
            nn_method = (
                "bruteforce"
                if self.dimension >= config.bruteforce_dimension_threshold
                else "kdtree"
            )
        return estimate_temperature(
            X,
            mode=config.scale_mode,
            c=c,
            max_points=config.max_points,
            random_state=self._next_seed() if random_seed is None else random_seed,
            nn_method=nn_method,
        )

    def _split_rejection_reason(self, result: SplitResult) -> str | None:
        values = (
            result.loss,
            result.parent_loss,
            result.split_gain,
            result.relative_split_gain,
        )
        if not np.all(np.isfinite(values)):
            return "nonfinite_split"
        if result.split_gain <= self.min_split_gain:
            return "insufficient_split_gain"
        if result.relative_split_gain <= self.min_relative_split_gain:
            return "insufficient_relative_split_gain"
        numerical_failures = {
            "line_search_failed",
            "max_line_search_failures",
            "alpha_min_reached",
            "non_descent_direction",
        }
        if result.stop_reason in numerical_failures and result.n_iters == 0:
            return "optimizer_failed_without_progress"
        return None

    def _split_warnings(self, result: SplitResult) -> tuple[str, ...]:
        warnings = []
        if not result.converged:
            warnings.append(result.stop_reason)
        if bool(result.metadata.get("alpha_min_saturated", False)):
            warnings.append("alpha_min_saturated")
        if bool(result.metadata.get("alpha_max_saturated", False)):
            warnings.append("alpha_max_saturated")
        return tuple(dict.fromkeys(warnings))

    def _split_summary(self, result: SplitResult) -> dict[str, object]:
        return {
            "split_mse": result.loss,
            "split_gain_mse": result.split_gain,
            "relative_split_gain_mse": result.relative_split_gain,
            "split_n_left": result.n_left,
            "split_n_right": result.n_right,
            "split_converged": result.converged,
            "split_stop_reason": result.stop_reason,
            "split_iterations": result.n_iters,
        }

    def _node_metadata(
        self,
        samples: _NodeSamples,
        fit_error: float,
        fit_mse: float,
    ) -> dict[str, object]:
        return {
            "fit_error": fit_error,
            "fit_error_metric": self._metric_name(),
            "fit_mse": fit_mse,
            "n_samples": samples.X.shape[0],
            "n_inherited": samples.n_inherited,
            "n_new": samples.n_new,
        }

    def _make_leaf(
        self,
        model: RegressionModel,
        region: PolytopeRegion,
        depth: int,
        node_id: str,
        status: str,
        metadata: dict[str, object],
        samples: _NodeSamples,
    ) -> LeafNode:
        metadata = dict(metadata)
        self._store_samples(metadata, samples)
        return LeafNode(
            model=model,
            region=region,
            depth=depth,
            node_id=node_id,
            status=status,
            metadata=metadata,
        )

    def _store_samples(
        self,
        metadata: dict[str, object],
        samples: _NodeSamples,
    ) -> None:
        if self.store_samples:
            metadata["X"] = samples.X.copy()
            metadata["y"] = samples.y.copy()

    def _sample_root(self, region: PolytopeRegion) -> np.ndarray:
        if isinstance(self.domain, BoxDomain) and self.exact_box_root:
            return sample_uniform_box(
                self.domain.bounds,
                self._target_samples,
                random_state=self._next_seed(),
            )
        X = np.asarray(
            self.root_sampler.sample(
                region,
                self._target_samples,
                random_state=self._next_seed(),
                x0=None,
            ),
            dtype=float,
        )
        if X.shape != (self._target_samples, self.dimension):
            raise RuntimeError(
                f"Root sampler returned shape {X.shape}; "
                f"expected {(self._target_samples, self.dimension)}."
            )
        return X

    def _query_oracle(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.oracle_vectorized:
            values = np.asarray(self.oracle(X), dtype=float).reshape(-1)
            if values.shape[0] != X.shape[0]:
                raise ValueError(
                    "A vectorized oracle must return one value per input row."
                )
            self.oracle_queries += X.shape[0]
            return values

        values = np.empty(X.shape[0], dtype=float)
        for index, x in enumerate(X):
            values[index] = float(self.oracle(x))
            self.oracle_queries += 1
        return values

    def _resolve_effective_dimension(self) -> int | None:
        if self.sample_count is not None:
            try:
                return model_effective_dimension(self.model_template, self.dimension)
            except TypeError:
                return None
        return model_effective_dimension(self.model_template, self.dimension)

    def _resolve_sample_count(self) -> int:
        if self.sample_count is None:
            return self.sample_multiplier * int(self._effective_dimension)
        if callable(self.sample_count):
            return int(self.sample_count(self.model_template, self.dimension))
        return int(self.sample_count)

    def _metric_name(self) -> str:
        return getattr(self.error_metric, "__name__", type(self.error_metric).__name__)

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**31 - 1))
