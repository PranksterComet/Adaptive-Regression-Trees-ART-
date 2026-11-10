"""GPU-resident model fits and projected descent for soft oblique splits.

The CPU controls stopping and adaptive step sizes. Only small status records
cross the device boundary during optimization. Numerical state stays on the GPU;
the returned SplitResult contains CPU models/diagnostics or a geometry-only
proposal when a builder will top up and fit its children.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np

from ._cuda_optimizer_kernels import kernels
from ._cuda_qr import QrPlan
from .models import (
    AffineRidgeModel, MODEL_CONDITION_WARNING_THRESHOLD, PolynomialRidgeModel,
    RidgeSolveResult, augment_features,
    ridge_solve_diagnostics,
)
from .optimizers import AdaptiveAlpha, LineSearchResult
from .splitters import SplitNotFoundError, SplitResult, _partition_result, project_unit_w


def validate_gpu_options(splitter) -> None:
    """Reject unsupported combinations before importing CuPy or fitting models."""
    backend = splitter.optimizer
    if backend == "hybrid_gram":
        raise ValueError("hybrid_gram uses the CPU optimizer and a CUDA Gram workspace.")
    if type(splitter.model_template) not in (AffineRidgeModel, PolynomialRidgeModel):
        raise TypeError(f"{backend} supports the built-in affine and polynomial ridge models.")
    if splitter.refit_during_line_search:
        raise ValueError(f"{backend} currently requires refit_during_line_search=False.")
    for name in ("temperature", "alpha0", "weight_floor"):
        value = getattr(splitter, name)
        if not np.isfinite(value) or value < 0 or (name != "weight_floor" and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'weight_floor' else 'positive'}.")
    if not np.isfinite(splitter.model_template.ridge) or splitter.model_template.ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    if not 0 < splitter.rho < 1 or not 0 < splitter.armijo_c < 1:
        raise ValueError("rho and armijo_c must lie strictly between 0 and 1.")
    for name, minimum in (("backtrack_batch_size", 1), ("Xw_refresh", 0), ("max_backtracks", 0)):
        value = getattr(splitter, name)
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}.")


validate_qr_options = validate_gpu_options  # Retain the existing internal entry point.


def _make_workspace(splitter, X, y, prepared_design, workspace_type):
    """Create a lazy workspace, reusable across restarts and same-node retries."""
    validate_gpu_options(splitter)
    model = splitter.model_template
    if type(model) is PolynomialRidgeModel:
        if prepared_design is None:
            prepared_design = model.prepare_design(X)
        model._validate_prepared(prepared_design)
        design = prepared_design.matrix
    else:
        design = augment_features(X)
    return workspace_type(X, y, design, model.ridge, splitter.weight_floor,
                          splitter.backtrack_batch_size, splitter.qr_parallel)


def make_qr_workspace(splitter, X, y, prepared_design=None):
    return _make_workspace(splitter, X, y, prepared_design, CudaQrWorkspace)


def make_cholesky_workspace(splitter, X, y, prepared_design=None):
    return _make_workspace(splitter, X, y, prepared_design, CudaCholeskyWorkspace)


class _Timers:
    """Deferred CUDA-event timings, including stream joins but no global fences."""

    def __init__(self, cp, profile, build_profile):
        self.cp, self.profile, self.build_profile = cp, profile, build_profile
        self.events = []

    @contextmanager
    def measure(self, category, count=0):
        if self.profile is None and self.build_profile is None:
            yield
            return
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((category, count, start, end))

    def flush(self):
        for category, count, start, end in self.events:
            end.synchronize()
            seconds = self.cp.cuda.get_elapsed_time(start, end) / 1000
            if self.profile is not None:
                name = f"{category}_seconds"
                setattr(self.profile, name, getattr(self.profile, name) + seconds)
                name = f"{category}_count"
                setattr(self.profile, name, getattr(self.profile, name) + count)
            if self.build_profile is not None and category == "least_squares":
                self.build_profile.optimizer_model_refit_seconds += seconds
        self.events.clear()


class CudaQrWorkspace:
    """Storage for one ordered dataset. A workspace is used serially."""

    backend = "cupy_qr"
    solver_requested = solver = "qr"
    label = "GPU QR"
    failure_reason = "qr_failure"

    def __init__(self, X, y, design, ridge, weight_floor, batch_size, parallel):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.design = np.asarray(design, dtype=float)
        self.n, self.d = self.X.shape
        self.p = self.design.shape[1]
        if self.n == 0 or self.d == 0 or self.p == 0:
            raise ValueError(f"{self.label} requires nonempty inputs.")
        if self.design.shape[0] != self.n or self.y.size != self.n:
            raise ValueError(f"{self.label} input dimensions do not match.")
        if not all(np.isfinite(a).all() for a in (self.X, self.y, self.design)):
            raise ValueError(f"{self.label} inputs must be finite.")
        self.ridge, self.weight_floor = float(ridge), float(weight_floor)
        self.batch_size, self.parallel = int(batch_size), bool(parallel)
        self.m = self.n + (self.p if ridge > 0 else 0)
        if self.solver == "qr" and self.m < self.p:
            raise ValueError("Unregularized GPU QR requires n >= p.")
        if max(self.n, self.d, self.p, self.m) > np.iinfo(np.int32).max:
            raise ValueError(f"{self.label} dimensions exceed the cuSOLVER integer limit.")
        self.ridge_scale = self.ridge / self.p
        self.blocks = (self.n + 255) // 256
        self.cp = None

    def validate(self, splitter, X, y, prepared_design):
        design = (prepared_design.matrix if type(splitter.model_template) is PolynomialRidgeModel
                  else augment_features(X))
        if (self.backend != splitter.optimizer
                or self.ridge != splitter.model_template.ridge
                or self.weight_floor != splitter.weight_floor
                or self.batch_size != splitter.backtrack_batch_size
                or self.parallel != splitter.qr_parallel
                or not np.array_equal(self.X, X)
                or not np.array_equal(self.y, y)
                or not np.array_equal(self.design, design)):
            raise ValueError(f"{self.label} workspace belongs to different data or configuration.")

    def initialize(self, profile, build_profile):
        if self.cp is not None:
            self.timers = _Timers(self.cp, profile, build_profile)
            return
        start = perf_counter()
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(f"optimizer='{self.backend}' requires CuPy and CUDA.") from exc
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.functions = kernels(cp.cuda.Device().id)
        with self.stream:
            self.x = cp.asarray(self.X, order="F")
            self.phi = cp.asarray(self.design, order="F")
            self.target = cp.asarray(self.y)
            self.energy = (cp.einsum("ij,ij->i", self.phi, self.phi)
                           if self.ridge > 0 or self.solver == "cholesky" else None)
            self.theta, self.last_feasible = cp.empty(self.d + 1), cp.empty(self.d + 1)
            self.u, self.v = cp.empty(self.n), cp.empty(self.n)
            self.pi = cp.empty(self.n)
            self.q, self.sq = cp.empty((2, self.n)), cp.empty((2, self.n))
            self.traces = cp.zeros(2)
            self.status = cp.empty(2, dtype=cp.int32)
            self.pred = cp.empty((self.n, 2), order="F")
            self.errors = cp.empty((2, self.n))
            self.coeff, self.gw = cp.empty(self.n), cp.empty(self.d)
            self.partials, self.totals = cp.empty((3, self.blocks)), cp.empty(3)
            self.pw = cp.empty(self.d + 1)
            self.geometry, self.packet = cp.empty(4), cp.empty(7)
            self.inv_r, self.response = cp.empty(1), cp.empty(3)
            self.trial_partials = cp.empty((self.batch_size, self.blocks))
            self.trial_values = cp.empty(self.batch_size)
            streams = ([cp.cuda.Stream(non_blocking=True) for _ in range(2)]
                       if self.parallel else [self.stream, self.stream])
            self.plans = self._make_plans(cp, streams)
        self.stream.synchronize()
        self.cp = cp
        self.timers = _Timers(cp, profile, build_profile)
        elapsed = perf_counter() - start
        if profile is not None:
            profile.least_squares_seconds += elapsed
        if build_profile is not None:
            build_profile.optimizer_model_refit_seconds += elapsed

    def _make_plans(self, cp, streams):
        return [QrPlan(cp, self.m, self.p, stream) for stream in streams]

    def call(self, name, grid, *args):
        # Raw kernels need explicit scalar widths; arrays are passed by device pointer.
        args = tuple(np.int32(a) if isinstance(a, (int, bool)) else
                     np.float64(a) if isinstance(a, float) else a for a in args)
        self.functions[name](grid, (256,), args)

    def event(self):
        event = self.cp.cuda.Event(disable_timing=True)
        event.record()
        return event

    def gates(self, temperature, *, alpha=0., advance=False, hard=False):
        self.hard_weights = hard
        with self.timers.measure("sigmoid", count=1):
            self.call("gates", (self.blocks,), self.n, self.d, temperature,
                      self.weight_floor, alpha, advance, hard, self.theta, self.v,
                      self.inv_r, self.u, self.pi, self.q, self.sq)

    def start(self, theta, temperature):
        self.host_theta = project_unit_w(theta)
        self.theta.set(self.host_theta)
        self.cp.matmul(self.x, self.theta[:-1], out=self.u)
        self.gates(temperature)

    def fit(self):
        """Fit and predict both sides, optionally overlapping independent streams."""
        cp = self.cp
        with self.timers.measure("least_squares", count=1):
            if self.energy is not None:
                cp.matmul(self.q, self.energy, out=self.traces)
            ready = self.event()
            done = []
            for side, plan in enumerate(self.plans):
                with plan.stream:
                    plan.stream.wait_event(ready)
                    self.call("qr_inputs", ((self.m * self.p + 255) // 256,),
                              self.n, self.p, self.m, side, self.ridge_scale,
                              self.phi, self.target, self.sq[side], self.traces,
                              plan.a, plan.b)
                    plan.solve()
                    # A diagonal threshold is a safety check, not a condition estimate
                    # or the rank-revealing test used by CPU pivoted QR.
                    self.call("qr_check", (1,), self.m, self.p, side,
                              float(np.finfo(float).eps * max(self.m, self.p)),
                              plan.a, plan.b, plan.info, self.status)
                    cp.matmul(self.phi, plan.coefficients, out=self.pred[:, side])
                    done.append(self.event())
            for event in done:
                self.stream.wait_event(event)

    def residuals(self, temperature):
        self.call("residual_stats", (self.blocks,), self.n, temperature,
                  self.target, self.pred, self.pi, self.errors, self.coeff, self.partials)

    def current(self, temperature):
        self.fit()
        with self.timers.measure("gradient", count=1):
            self.residuals(temperature)
            ready = self.event()
            # The GEMV and the scalar reductions share no output dependencies.
            stream = self.plans[0].stream
            with stream:
                stream.wait_event(ready)
                self.cp.matmul(self.x.T, self.coeff, out=self.gw)
                done = self.event()
            self.cp.sum(self.partials, axis=1, out=self.totals)
            self.stream.wait_event(done)
            self.call("direction", (1,), self.d, self.ridge_scale, self.theta,
                      self.gw, self.totals, self.traces, self.status, self.pw,
                      self.geometry, self.packet)
        packet = self.packet.get(stream=self.stream)
        if packet[6]:
            message = ("GPU QR failed or detected numerical rank loss." if self.solver == "qr"
                       else "GPU Cholesky failed or produced nonfinite results.")
            raise SplitNotFoundError(self.failure_reason,
                                     message,
                                     diagnostics={"optimizer": self.backend})
        return packet

    def search(self, splitter, alpha_start):
        cp = self.cp
        steps = np.empty(splitter.max_backtracks + 1)
        steps[0] = alpha_start
        for i in range(1, steps.size):
            steps[i] = steps[i - 1] * splitter.rho
        alphas = cp.asarray(steps)
        physical = logical = 0
        with self.timers.measure("backtracking", count=1):
            for offset in range(0, steps.size, self.batch_size):
                count = min(self.batch_size, steps.size - offset)
                with self.timers.measure("sigmoid", count=count):
                    self.call("trial_losses", (self.blocks, count), self.n, self.d,
                              offset, float(splitter.temperature), self.theta, self.u,
                              self.v, self.errors, self.geometry, alphas, self.trial_partials)
                    cp.sum(self.trial_partials[:count], axis=1, out=self.trial_values[:count])
                self.call("select_trial", (1,), count, offset, float(splitter.armijo_c),
                          self.trial_values, alphas, self.packet, self.response)
                selected, value, alpha = self.response.get(stream=self.stream)
                physical += count
                logical += int(selected) + 1 if selected >= 0 else count
                if selected >= 0:
                    break
        if self.timers.profile is not None:
            self.timers.profile.line_search_candidate_count += physical
        # No theta is transferred: the controller uses only these scalar fields.
        result = LineSearchResult(np.empty(0), float(value), float(alpha),
                                  bool(selected >= 0), logical - 1)
        return result, physical, logical

    def advance(self, alpha, temperature, refresh):
        self.call("commit_theta", (1,), self.d, float(alpha), self.pw,
                  self.geometry, self.theta, self.inv_r)
        if refresh:
            self.cp.matmul(self.x, self.theta[:-1], out=self.u)
        self.gates(temperature, alpha=float(alpha), advance=not refresh)

    def gate_summary(self):
        cp = self.cp
        stats = cp.stack((cp.mean(self.pi), cp.min(self.pi), cp.max(self.pi)))
        mean, low, high = stats.get(stream=self.stream)
        return {"mean_pi": float(mean), "min_pi": float(low), "max_pi": float(high)}

    def hard_result(self, splitter, parent_loss, required, converged, metadata, prepared,
                    *, hard_refit=True):
        cp = self.cp
        # Recompute scores for the final hard routing; soft iterations use cached scores.
        self.theta[:-1] /= cp.linalg.norm(self.theta[:-1])
        cp.matmul(self.x, self.theta[:-1], out=self.u)
        self.gates(float(splitter.temperature), hard=True)
        n_right = int(cp.sum(self.pi).get(stream=self.stream))
        n_left = self.n - n_right
        theta = self.theta.get(stream=self.stream)
        if min(n_left, n_right) < required:
            raise SplitNotFoundError("min_side_points", f"{self.label} split is below the minimum side count.",
                diagnostics={**metadata, "final_theta": theta, "required_side_points": required,
                             "n_left": n_left, "n_right": n_right})
        metadata.update(hard_refit=hard_refit, n_left=n_left, n_right=n_right,
                        required_side_points=required)
        if not hard_refit:
            return _partition_result(theta, self.n, n_right, parent_loss, converged, metadata)
        self.fit()
        self.residuals(float(splitter.temperature))
        cp.sum(self.partials, axis=1, out=self.totals)
        status = self.status.get(stream=self.stream)
        if np.any(status):
            raise SplitNotFoundError(self.failure_reason, f"Hard {self.label} fit failed numerical checks.",
                                     diagnostics=metadata)
        loss = float(self.totals[0].get(stream=self.stream))
        side_sums = cp.sum(self.errors * self.q, axis=1).get(stream=self.stream)
        ridges = (self.ridge_scale * self.traces).get(stream=self.stream)
        if not np.isfinite(loss) or not np.isfinite(side_sums).all():
            raise SplitNotFoundError(self.failure_reason, f"Hard {self.label} fit produced a nonfinite loss.",
                                     diagnostics=metadata)
        models = []
        for side, plan in enumerate(self.plans):
            model = splitter.model_template.clone()
            model.coef_ = plan.coefficients.get(stream=self.stream)
            if type(model) is PolynomialRidgeModel:
                model.transformer_ = prepared.feature_state
            model.solve_result_ = RidgeSolveResult(
                coefficients=model.coef_, solver_requested=self.solver_requested, solver_used=self.solver,
                condition_estimator=None, cond_estimate=None,
                ridge_effective=float(ridges[side]), rank=None,
            )
            models.append(model)
        metadata.update(required_side_points=required, left_loss=float(side_sums[0] / n_left),
                        right_loss=float(side_sums[1] / n_right),
                        hard_left_solve=ridge_solve_diagnostics(models[0]),
                        hard_right_solve=ridge_solve_diagnostics(models[1]))
        return SplitResult(
            w=theta[:-1], z=float(theta[-1]), left_model=models[0], right_model=models[1],
            loss=loss, parent_loss=parent_loss, split_gain=parent_loss - loss,
            relative_split_gain=(parent_loss - loss) / max(parent_loss, 1e-12),
            n_left=n_left, n_right=n_right, converged=converged,
            n_iters=metadata["n_iters"], stop_reason=metadata["stop_reason"], metadata=metadata,
        )


class CudaCholeskyWorkspace(CudaQrWorkspace):
    """Reuse the resident optimizer with complementary Gram/Cholesky fits."""

    backend = "cupy_cholesky"
    solver_requested = "normal"
    solver = "cholesky"
    label = "GPU Cholesky"
    failure_reason = "cholesky_failure"

    def _make_plans(self, cp, streams):
        from ._cuda_cholesky import CholeskyPlan, ComplementaryGrams

        self.gram_pair = ComplementaryGrams(cp, self.phi, self.target)
        return [CholeskyPlan(cp, self.p, stream) for stream in streams]

    def fit(self):
        cp = self.cp
        with self.timers.measure("least_squares", count=1):
            cp.matmul(self.q, self.energy, out=self.traces)
            self.gram_pair.assemble(self.plans, self.q, self.pi, self.traces,
                                    0.0 if self.hard_weights else self.weight_floor, self.stream)
            for side, plan in enumerate(self.plans):
                self.call("add_ridge", ((self.p + 255) // 256,),
                          self.p, side, self.ridge_scale, self.traces, plan.a)
            # Derive both Grams before either plan overwrites its input factors.
            ready = self.event()
            done = []
            for side, plan in enumerate(self.plans):
                with plan.stream:
                    plan.stream.wait_event(ready)
                    plan.solve()
                    self.call("cholesky_check", (1,), self.p, side,
                              plan.a, plan.b, plan.info, self.status)
                    cp.matmul(self.phi, plan.coefficients, out=self.pred[:, side])
                    done.append(self.event())
            for event in done:
                self.stream.wait_event(event)


def _optimize(splitter, ws, theta0, parent_loss, required, parent_solve, prepared, *, hard_refit=True):
    solver = getattr(ws, "solver", "qr")
    solver_requested = getattr(ws, "solver_requested", "qr")
    ws.start(theta0, float(splitter.temperature))
    current = ws.current(float(splitter.temperature))
    losses, norms, steps, backtracks, trial_losses = [float(current[0])], [float(current[1])], [], [], []
    tolerance = splitter.grad_atol + splitter.grad_rtol * norms[0]
    controller = AdaptiveAlpha(splitter.alpha0, splitter.alpha_min, splitter.alpha_max,
                               splitter.alpha_grow, splitter.alpha_recovery,
                               splitter.heavy_backtrack_threshold)
    last_feasible = None
    physical = logical = 0
    converged, reason = False, "max_iters"

    def save_feasible(iteration):
        nonlocal last_feasible
        if min(current[3], ws.n - current[3]) >= required:
            ws.cp.copyto(ws.last_feasible, ws.theta)
            last_feasible = iteration

    save_feasible(0)
    for _ in range(splitter.max_iters):
        if current[1] <= tolerance:
            converged, reason = True, "gradient_tolerance"
            break
        if current[2] >= 0:
            reason = "non_descent_direction"
            break
        with ws.timers.measure("gradient"):
            ws.cp.matmul(ws.x, ws.pw[:-1], out=ws.v)
        failures = 0
        while True:
            alpha = controller.alpha if splitter.adaptive_alpha else splitter.alpha0
            search, actual, examined = ws.search(splitter, alpha)
            physical += actual
            logical += examined
            if splitter.adaptive_alpha:
                controller.update(search, splitter.rho)
            if search.success:
                break
            failures += 1
            if not splitter.adaptive_alpha:
                reason = "line_search_failed"
                break
            if failures >= splitter.max_line_search_failures:
                reason = "max_line_search_failures"
                break
            if controller.alpha <= splitter.alpha_min:
                reason = "alpha_min_reached"
                break
        if not search.success:
            break
        iteration = len(steps) + 1
        refresh = splitter.Xw_refresh > 0 and iteration % splitter.Xw_refresh == 0
        ws.advance(search.step_size, float(splitter.temperature), refresh)
        current = ws.current(float(splitter.temperature))
        losses.append(float(current[0]))
        norms.append(float(current[1]))
        steps.append(search.step_size)
        backtracks.append(search.n_backtracks)
        trial_losses.append(search.value)
        save_feasible(iteration)

    def solve_metadata(side):
        return {"solver_requested": solver_requested, "solver_used": solver, "cond_estimate": None,
                "condition_estimator": None, "rank": None, "fallback_reason": None,
                "ridge_effective": float(current[4 + side])}

    metadata = {
        "optimizer": getattr(ws, "backend", "cupy_qr"),
        "fit_parallel": splitter.qr_parallel,
        "gpu_solver": solver, "solver_fallback": False, "condition_estimation": False,
        "backtrack_batch_size": splitter.backtrack_batch_size, "Xw_refresh": splitter.Xw_refresh,
        "timing_clock": "cuda_events_with_wall_time_setup",
        "physical_line_search_candidates": physical, "logical_line_search_candidates": logical,
        "speculative_line_search_candidates": physical - logical,
        "soft_loss_history": losses, "line_search_loss_history": trial_losses,
        "projected_grad_norm_history": norms,
        "left_cond_estimate_history": [None] * len(losses),
        "right_cond_estimate_history": [None] * len(losses),
        "left_condition_estimator_history": [None] * len(losses),
        "right_condition_estimator_history": [None] * len(losses),
        "left_solver_history": [solver] * len(losses), "right_solver_history": [solver] * len(losses),
        "step_size_history": steps, "backtrack_history": backtracks,
        "stop_reason": reason, "temperature": splitter.temperature,
        "adaptive_alpha": splitter.adaptive_alpha, "refit_during_line_search": False,
        "initial_grad_norm": norms[0], "final_grad_norm": norms[-1], "n_iters": len(steps),
        "grad_tolerance": tolerance,
        "final_alpha": controller.alpha if splitter.adaptive_alpha else splitter.alpha0,
        "alpha_min_saturated": bool(splitter.adaptive_alpha and controller.alpha <= splitter.alpha_min),
        "alpha_max_saturated": bool(splitter.adaptive_alpha and controller.alpha >= splitter.alpha_max),
        "final_soft_metadata": {
            "temperature": splitter.temperature, "model_type": type(splitter.model_template).__name__,
            **ws.gate_summary(), "models_refit": True, "prepared_design": prepared is not None,
            "left_solve": solve_metadata(0), "right_solve": solve_metadata(1),
        },
        "parent_solve": parent_solve,
        # These GPU backends do not provide a condition estimate.
        "high_model_conditioning": bool(parent_solve and parent_solve.get("cond_estimate") is not None
                                        and parent_solve["cond_estimate"] >= MODEL_CONDITION_WARNING_THRESHOLD),
    }
    if solver == "qr":
        metadata.update(qr_parallel=splitter.qr_parallel, qr_pivoted=False)
    else:
        metadata.update(cholesky_parallel=splitter.qr_parallel, complementary_grams=True,
                        gram_floor_correction=True, rhs_assembly="direct")
    if (min(current[3], ws.n - current[3]) < required and splitter.min_side_fallback
            and last_feasible is not None):
        ws.cp.copyto(ws.theta, ws.last_feasible)
        metadata.update(min_side_fallback=True, fallback_iteration=last_feasible)
    options = {} if hard_refit else {"hard_refit": False}
    return ws.hard_result(splitter, parent_loss, required, converged, metadata, prepared, **options)


def split_cuda_qr(splitter, X, y, rng, parent_loss, required, parent_solve,
                  prepared_design=None, workspace=None, *, hard_refit=True):
    return _split_cuda_optimizer(splitter, X, y, rng, parent_loss, required, parent_solve,
                                 prepared_design, workspace, make_qr_workspace, hard_refit=hard_refit)


def split_cuda_cholesky(splitter, X, y, rng, parent_loss, required, parent_solve,
                         prepared_design=None, workspace=None, *, hard_refit=True):
    return _split_cuda_optimizer(splitter, X, y, rng, parent_loss, required, parent_solve,
                                 prepared_design, workspace, make_cholesky_workspace, hard_refit=hard_refit)


def _split_cuda_optimizer(splitter, X, y, rng, parent_loss, required, parent_solve,
                           prepared_design, workspace, factory, *, hard_refit=True):
    """Run independent restarts without retaining a device object in the result."""
    ws = workspace if workspace is not None else factory(splitter, X, y, prepared_design)
    ws.validate(splitter, X, y, prepared_design)
    ws.initialize(splitter.split_timing_profile, splitter._timing_profile)
    best = fallback = last_failure = None
    try:
        with ws.stream:
            for _ in range(max(1, splitter.n_restarts)):
                theta0 = splitter._initialize_theta(X, rng)
                try:
                    result = _optimize(splitter, ws, theta0, parent_loss, required,
                                       parent_solve, prepared_design, hard_refit=hard_refit)
                except SplitNotFoundError as exc:
                    last_failure = exc
                    continue
                if result.metadata.get("min_side_fallback", False):
                    if fallback is None or result.loss < fallback.loss:
                        fallback = result
                elif best is None or result.loss < best.loss:
                    best = result
        if best is not None:
            return best
        if fallback is not None:
            return fallback
        raise last_failure or SplitNotFoundError("invalid_split", f"{ws.label} found no valid split.")
    finally:
        # Also drain child streams if a library call raised before the normal join.
        for plan in ws.plans:
            plan.stream.synchronize()
        ws.stream.synchronize()
        ws.timers.flush()

