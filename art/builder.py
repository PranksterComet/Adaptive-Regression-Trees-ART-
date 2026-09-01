"""Recursive construction of adaptive regression trees."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable

import numpy as np

from .domain import BoxDomain, PolytopeRegion, split_region
from .metrics import mean_squared_error, relative_l2_error
from .models import (
    AffineRidgeModel,
    MODEL_CONDITION_WARNING_THRESHOLD,
    PreparedDesign,
    PreparedFeatureModel,
    RegressionModel,
    WeightedRegressionModel,
    model_effective_dimension,
    ridge_solve_diagnostics,
)
from .sampling import (
    HitAndRunSampler,
    Sampler,
    floor_covariance_eigenvalues,
    sample_covariance_eigendecomposition,
    sample_uniform_box,
)
from .splitters import (
    HingeAffineSplitter,
    SoftObliqueSplitter,
    SplitNotFoundError,
    SplitResult,
    Splitter,
)
from .temperature import TemperatureConfig, estimate_temperature
from .timing import BuildTimingCategory, BuildTimingProfile
from .tree import LeafNode, RegressionTree, SplitNode, TreeNode


ErrorMetric = Callable[[np.ndarray, np.ndarray], float]
SampleCountPolicy = Callable[[RegressionModel, int], int]


@dataclass(frozen=True)
class TreeBuildResult:
    tree: RegressionTree
    oracle_queries: int
    restarts_on_failure: int
    build_timing: dict[str, float] | None = None


@dataclass
class _NodeSamples:
    X: np.ndarray
    y: np.ndarray
    n_inherited: int
    n_new: int
    sampling_metadata: dict[str, object] = field(default_factory=dict)


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
    isotropic_sampling: bool = True
    isotropic_pilot_multiplier: int = 50
    temperature_config: TemperatureConfig | None = None
    min_split_gain: float = 0.0
    min_relative_split_gain: float = 0.0
    max_retries_on_failure: int = 0
    store_samples: bool = False
    store_diagnostics: bool = False
    profile_build_timing: bool = False
    oracle_vectorized: bool = False
    random_state: int | np.random.Generator | None = None

    oracle_queries: int = field(default=0, init=False)
    restarts_on_failure: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _root_temperature_c: float | None = field(default=None, init=False, repr=False)
    _target_samples: int = field(init=False, repr=False)
    _effective_dimension: int | None = field(init=False, repr=False)
    _timing_profile: BuildTimingProfile | None = field(
        default=None,
        init=False,
        repr=False,
    )

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
        if self.isotropic_sampling and self.isotropic_pilot_multiplier < 2:
            raise ValueError("isotropic_pilot_multiplier must be at least 2.")
        if self.isotropic_sampling and not isinstance(self.sampler, HitAndRunSampler):
            raise TypeError("isotropic_sampling requires a HitAndRunSampler.")
        root_uses_hit_and_run = not (
            isinstance(self.domain, BoxDomain) and self.exact_box_root
        )
        if (
            self.isotropic_sampling
            and root_uses_hit_and_run
            and not isinstance(self.root_sampler, HitAndRunSampler)
        ):
            raise TypeError(
                "isotropic_sampling requires a HitAndRunSampler for non-exact roots."
            )
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

        self._timing_profile = (
            BuildTimingProfile() if self.profile_build_timing else None
        )
        build_start = perf_counter() if self._timing_profile is not None else None
        self.oracle_queries = 0
        self.restarts_on_failure = 0
        self._root_temperature_c = None
        root_region = self.domain.as_region() if isinstance(self.domain, BoxDomain) else self.domain
        X_root, sampling_metadata = self._sample_root(root_region)
        y_root = self._query_oracle(X_root)
        root_samples = _NodeSamples(
            X=X_root,
            y=y_root,
            n_inherited=0,
            n_new=X_root.shape[0],
            sampling_metadata=sampling_metadata,
        )
        root = self._build_node(root_region, root_samples, depth=0, node_id="root")
        tree = RegressionTree(
            root=root,
            oracle_queries=self.oracle_queries,
            metadata={
                "error_metric": self._metric_name(),
                "error_threshold": self.error_threshold,
                "effective_dimension": self._effective_dimension,
                "splitter": type(self.splitter).__name__,
                "target_samples_per_node": self._target_samples,
                "restarts_on_failure": self.restarts_on_failure,
                "isotropic_sampling": self.isotropic_sampling,
                "isotropic_pilot_multiplier": self.isotropic_pilot_multiplier,
                "temperature_strategy": (
                    None if self.temperature_config is None else self.temperature_config.strategy
                ),
            },
        )
        build_timing = None
        if self._timing_profile is not None and build_start is not None:
            build_timing = self._timing_profile.summary(perf_counter() - build_start)
        tree.metadata["build_timing"] = build_timing
        return TreeBuildResult(
            tree=tree,
            oracle_queries=self.oracle_queries,
            restarts_on_failure=self.restarts_on_failure,
            build_timing=build_timing,
        )

    def _build_node(
        self,
        region: PolytopeRegion,
        samples: _NodeSamples,
        depth: int,
        node_id: str,
    ) -> TreeNode:
        prepared_design = self._prepare_design(samples.X)
        model, predictions = self._fit_node_model(
            samples.X,
            samples.y,
            prepared_design,
        )
        fit_error = float(self.error_metric(samples.y, predictions))
        parent_mse = mean_squared_error(samples.y, predictions)
        base_metadata = self._node_metadata(samples, fit_error, parent_mse, model)

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
                prepared_design,
            )
        except SplitNotFoundError as exc:
            warnings = self._combine_warnings(
                base_metadata.get("warnings", ()),
                self._diagnostic_warnings(exc.diagnostics),
            )
            metadata = {
                **base_metadata,
                **exc.context,
                "split_failure_reason": exc.reason,
                "restarts_on_failure": exc.restarts_on_failure,
                "split_attempt_failure_reasons": exc.failure_reasons,
                "warnings": warnings,
            }
            if self.store_diagnostics and exc.diagnostics is not None:
                metadata["splitter_metadata"] = copy.deepcopy(exc.diagnostics)
            return self._make_leaf(
                model, region, depth, node_id, exc.reason, metadata, samples
            )

        rejection_reason = self._split_rejection_reason(split_result)
        if rejection_reason is not None:
            warnings = self._combine_warnings(
                base_metadata.get("warnings", ()),
                self._split_warnings(split_result),
            )
            metadata = {
                **base_metadata,
                **self._split_summary(split_result),
                **temperature_metadata,
                **retry_metadata,
                "warnings": warnings,
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
        del prepared_design
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

        warnings = self._combine_warnings(
            base_metadata.get("warnings", ()),
            self._split_warnings(split_result),
        )
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
            prepared_design = self._prepare_design(inherited_X)
            model, predictions = self._fit_node_model(
                inherited_X,
                inherited_y,
                prepared_design,
            )
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
                model,
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
            return _NodeSamples(
                inherited_X,
                inherited_y,
                n_inherited,
                0,
                sampling_metadata={
                    "sampling_method": "inherited_only",
                    "sampling_thinning": None,
                    "sampling_pilot_length": 0,
                    "sampling_covariance_condition_number": None,
                    "sampling_warnings": (),
                },
            )

        x0 = inherited_X[int(self._rng.integers(0, n_inherited))]
        X_new, sampling_metadata = self._sample_region(
            region,
            n_new,
            x0=x0,
            sampler=self.sampler,
        )
        y_new = self._query_oracle(X_new)
        return _NodeSamples(
            X=np.vstack([inherited_X, X_new]),
            y=np.concatenate([inherited_y, y_new]),
            n_inherited=n_inherited,
            n_new=n_new,
            sampling_metadata=sampling_metadata,
        )

    def _fit_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_model: RegressionModel,
        parent_mse: float,
        depth: int,
        prepared_design: PreparedDesign | None,
    ) -> tuple[SplitResult, dict[str, object], dict[str, object]]:
        temperature, temperature_metadata, temperature_tuned = (
            self._resolve_split_temperature(X, y, depth, prepared_design)
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
                    prepared_design,
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
        prepared_design: PreparedDesign | None,
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
            c, candidate_scores = self._tune_temperature(X, y, prepared_design)
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
        prepared_design: PreparedDesign | None,
    ) -> tuple[float, list[dict[str, object]]]:
        config = self.temperature_config
        n_val = max(1, int(round(config.validation_fraction * X.shape[0])))
        permutation = self._rng.permutation(X.shape[0])
        val_idx = permutation[:n_val]
        fit_idx = permutation[n_val:]
        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        fit_design = (
            None if prepared_design is None else prepared_design.subset(fit_idx)
        )
        validation_design = (
            None if prepared_design is None else prepared_design.subset(val_idx)
        )
        parent_model, parent_predictions = self._fit_node_model(
            X_fit,
            y_fit,
            fit_design,
        )
        parent_mse = mean_squared_error(y_fit, parent_predictions)
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
                    fit_design,
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

            validation_predictions = (
                result.predict(X_val)
                if validation_design is None
                else result.predict_prepared(X_val, validation_design)
            )
            validation_mse = mean_squared_error(y_val, validation_predictions)
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
        prepared_design: PreparedDesign | None,
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
        if isinstance(node_splitter, SoftObliqueSplitter):
            node_splitter._timing_profile = self._timing_profile
        split_kwargs: dict[str, object] = {
            "parent_model": parent_model,
            "parent_loss": parent_mse,
        }
        if prepared_design is not None:
            split_kwargs["prepared_design"] = prepared_design
        with self._timing_context("splitter"):
            return node_splitter.split(X, y, **split_kwargs)

    def _prepare_design(self, X: np.ndarray) -> PreparedDesign | None:
        if not isinstance(self.model_template, PreparedFeatureModel):
            return None
        return self.model_template.prepare_design(X)

    def _fit_node_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prepared_design: PreparedDesign | None,
    ) -> tuple[RegressionModel, np.ndarray]:
        model = self.model_template.clone()
        if prepared_design is None:
            model.fit(X, y)
            return model, model.predict(X)
        if not isinstance(model, PreparedFeatureModel):
            raise TypeError(
                "A prepared design requires clones implementing PreparedFeatureModel."
            )
        model.fit_design(prepared_design, y)
        return model, model.predict_design(prepared_design)

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
        if result.n_iters == 0 and result.stop_reason == "gradient_tolerance":
            warnings.append("small_init_grad")
        if not result.converged:
            warnings.append(result.stop_reason)
        if bool(result.metadata.get("alpha_min_saturated", False)):
            warnings.append("alpha_min_saturated")
        if bool(result.metadata.get("alpha_max_saturated", False)):
            warnings.append("alpha_max_saturated")
        if bool(result.metadata.get("high_model_conditioning", False)):
            warnings.append("high_model_conditioning")
        return tuple(dict.fromkeys(warnings))

    def _diagnostic_warnings(
        self,
        diagnostics: dict[str, object] | None,
    ) -> tuple[str, ...]:
        if diagnostics is None:
            return ()
        warnings = []
        if (
            diagnostics.get("n_iters") == 0
            and diagnostics.get("stop_reason") == "gradient_tolerance"
        ):
            warnings.append("small_init_grad")
        if bool(diagnostics.get("high_model_conditioning", False)):
            warnings.append("high_model_conditioning")
        return tuple(warnings)

    @staticmethod
    def _combine_warnings(*groups: object) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                warning
                for group in groups
                for warning in group
            )
        )

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
        model: RegressionModel,
    ) -> dict[str, object]:
        solve = ridge_solve_diagnostics(model)
        cond_estimate = None if solve is None else solve.get("cond_estimate")
        warnings = (
            ("high_model_conditioning",)
            if cond_estimate is not None
            and float(cond_estimate) >= MODEL_CONDITION_WARNING_THRESHOLD
            else ()
        )
        return {
            "fit_error": fit_error,
            "fit_error_metric": self._metric_name(),
            "fit_mse": fit_mse,
            "n_samples": samples.X.shape[0],
            "n_inherited": samples.n_inherited,
            "n_new": samples.n_new,
            "fit_solver_requested": (
                None if solve is None else solve.get("solver_requested")
            ),
            "fit_solver_used": None if solve is None else solve.get("solver_used"),
            "fit_condition_estimator": (
                None if solve is None else solve.get("condition_estimator")
            ),
            "fit_cond_estimate": cond_estimate,
            "fit_ridge_effective": (
                None if solve is None else solve.get("ridge_effective")
            ),
            "fit_rank": None if solve is None else solve.get("rank"),
            "fit_fallback_reason": (
                None if solve is None else solve.get("fallback_reason")
            ),
            "warnings": warnings,
            **samples.sampling_metadata,
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

    def _sample_root(
        self,
        region: PolytopeRegion,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if isinstance(self.domain, BoxDomain) and self.exact_box_root:
            with self._timing_context("sampling"):
                X = sample_uniform_box(
                    self.domain.bounds,
                    self._target_samples,
                    random_state=self._next_seed(),
                )
            return (
                X,
                {
                    "sampling_method": "exact_uniform_box",
                    "sampling_thinning": None,
                    "sampling_pilot_length": 0,
                    "sampling_covariance_condition_number": None,
                    "sampling_warnings": (),
                },
            )
        return self._sample_region(
            region,
            self._target_samples,
            x0=None,
            sampler=self.root_sampler,
        )

    def _sample_region(
        self,
        region: PolytopeRegion,
        n: int,
        *,
        x0: np.ndarray | None,
        sampler: Sampler,
    ) -> tuple[np.ndarray, dict[str, object]]:
        with self._timing_context("sampling"):
            return self._sample_region_untimed(
                region,
                n,
                x0=x0,
                sampler=sampler,
            )

    def _sample_region_untimed(
        self,
        region: PolytopeRegion,
        n: int,
        *,
        x0: np.ndarray | None,
        sampler: Sampler,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if not self.isotropic_sampling:
            X = np.asarray(
                sampler.sample(
                    region,
                    n,
                    random_state=self._next_seed(),
                    x0=x0,
                ),
                dtype=float,
            )
            metadata = {
                "sampling_method": type(sampler).__name__,
                "sampling_thinning": getattr(sampler, "thinning", None),
                "sampling_pilot_length": 0,
                "sampling_covariance_condition_number": None,
                "sampling_warnings": (),
            }
        else:
            if not isinstance(sampler, HitAndRunSampler):
                raise TypeError("isotropic_sampling requires a HitAndRunSampler.")

            pilot_length = self.isotropic_pilot_multiplier * self.dimension
            pilot_sampler = replace(
                sampler,
                burn_in=0,
                thinning=1,
                direction_eigenvectors=None,
                direction_eigenvalues=None,
            )
            pilot = np.asarray(
                pilot_sampler.sample(
                    region,
                    pilot_length,
                    random_state=self._next_seed(),
                    x0=x0,
                ),
                dtype=float,
            )
            if pilot.shape != (pilot_length, self.dimension):
                raise RuntimeError(
                    f"Pilot sampler returned shape {pilot.shape}; "
                    f"expected {(pilot_length, self.dimension)}."
                )

            eigenvectors, eigenvalues = sample_covariance_eigendecomposition(pilot)
            safe_eigenvalues, eigenvalue_floor = floor_covariance_eigenvalues(
                eigenvalues,
                sampler.direction_eigenvalue_floor,
            )
            floor_saturated = bool(np.any(eigenvalues <= eigenvalue_floor))
            condition_number = float(
                np.max(safe_eigenvalues) / np.min(safe_eigenvalues)
            )
            node_sampler = replace(
                sampler,
                direction_eigenvectors=eigenvectors,
                direction_eigenvalues=eigenvalues,
            )
            X = np.asarray(
                node_sampler.sample(
                    region,
                    n,
                    random_state=self._next_seed(),
                    x0=pilot[-1],
                ),
                dtype=float,
            )
            warning = "covariance_eigenvalue_floor_saturated"
            metadata = {
                "sampling_method": "isotropic_hit_and_run",
                "sampling_thinning": sampler.thinning,
                "sampling_pilot_length": pilot_length,
                "sampling_covariance_condition_number": condition_number,
                "sampling_covariance_eigenvalue_floor": eigenvalue_floor,
                "sampling_covariance_floor_saturated": floor_saturated,
                "sampling_warnings": (warning,) if floor_saturated else (),
            }

        if X.shape != (n, self.dimension):
            raise RuntimeError(
                f"Sampler returned shape {X.shape}; expected {(n, self.dimension)}."
            )
        if not np.all(region.contains(X)):
            raise RuntimeError("Sampler returned points outside the region.")
        return X, metadata

    def _timing_context(self, category: BuildTimingCategory):
        if self._timing_profile is None:
            return nullcontext()
        return self._timing_profile.measure(category)

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
