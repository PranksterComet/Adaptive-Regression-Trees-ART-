"""Reusable GPU Cholesky optimization with independent per-node GD state.

The builder supplies equally sized NumPy datasets. Large arrays stay in fixed
GPU slots; the CPU reads compact control records and owns adaptive-alpha/history
bookkeeping. Tree scheduling is deliberately outside this module.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ._cuda_batched_cholesky import BatchedCholeskyWorkspace, _dimensions, _node_ids
from ._cuda_batched_matmul import BatchedMatmul
from ._cuda_features import FeaturePlan
from ._cuda_optimizer_kernels import kernels
from .models import AffineRidgeModel, PolynomialRidgeModel, RidgeSolveResult
from .optimizers import AdaptiveAlpha, LineSearchResult
from .splitters import SplitNotFoundError, SplitResult, _partition_result, project_unit_w


@lru_cache(maxsize=None)
def _bookkeeping(device_id):
    import cupy as cp

    names = ("alpha_steps", "search_state", "save_feasible", "copy_theta",
             "normalize_theta", "hard_stats", "keep_best", "initial_directions",
             "initial_scores", "initial_offsets", "partition_counts")
    module = cp.RawModule(code=Path(__file__).with_suffix(".cu").read_text(),
                          options=("--std=c++11",), name_expressions=names)
    return {name: module.get_function(name) for name in names}


def _partition(ids, keep):
    """Swap removed IDs to the tail; large arrays keep their original slots."""
    i, count = 0, len(ids)
    while i < count:
        if keep[ids[i]]:
            i += 1
        else:
            count -= 1
            ids[i], ids[count] = ids[count], ids[i]
    return count


def _batch_data(X, y, temperature):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 3 or min(X.shape) == 0 or y.shape != X.shape[:2]:
        raise ValueError("X/y must have nonempty shapes (K,n,d)/(K,n).")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite.")
    temperature = np.broadcast_to(np.asarray(temperature, dtype=float), (X.shape[0],))
    if not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("Temperatures must be finite and positive.")
    _dimensions(*X.shape)
    return X, y, temperature


def _validate_settings(s, d, reorder):
    if type(s.model_template) not in (AffineRidgeModel, PolynomialRidgeModel):
        raise TypeError("Batched Cholesky supports affine and polynomial ridge models.")
    if type(s.model_template) is PolynomialRidgeModel and (
            not isinstance(s.model_template.degree, (int, np.integer))
            or s.model_template.degree not in (1, 2)):
        raise ValueError("Batched GPU features support only polynomial degree 1 or 2.")
    if s.optimizer == "hybrid_gram" or s.refit_during_line_search:
        raise ValueError("Batched optimization requires refit_during_line_search=False and a resident optimizer.")
    if not isinstance(reorder, (bool, np.bool_)):
        raise ValueError("reorder must be a boolean.")
    # This checks splitter options and dimensions without allocating a dataset.
    s._validate_inputs(np.empty((0, d)), np.empty(0))
    for name in ("alpha0", "alpha_min", "alpha_max", "grad_atol", "grad_rtol", "weight_floor"):
        value = getattr(s, name)
        if not np.isfinite(value) or value < 0 or (name.startswith("alpha") and value == 0):
            raise ValueError(f"Invalid {name}.")
    for name, minimum in (("max_iters", 0), ("max_backtracks", 0), ("Xw_refresh", 0),
                          ("backtrack_batch_size", 1), ("max_line_search_failures", 1), ("n_restarts", 0)):
        if not isinstance(getattr(s, name), (int, np.integer)) or getattr(s, name) < minimum:
            raise ValueError(f"Invalid {name}.")
    if not 0 < s.rho < 1 or not 0 < s.armijo_c < 1:
        raise ValueError("rho and armijo_c must lie strictly between 0 and 1.")
    if not np.isfinite(s.model_template.ridge) or s.model_template.ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    AdaptiveAlpha(s.alpha0, s.alpha_min, s.alpha_max, s.alpha_grow,
                  s.alpha_recovery, s.heavy_backtrack_threshold)


class BatchedOptimizerWorkspace:
    """Allocate capacity K once; load and optimize cohorts of k<=K nodes.

    X/y are NumPy (k,n,d)/(k,n); temperatures may vary by node. reorder=True
    removes completed nodes from GEMM/Cholesky pointer prefixes. False keeps
    all requested nodes in least-squares calls while other stages mask them.
    optimize() returns a list in requested slot order (all k by default), each
    entry a SplitResult or SplitNotFoundError. Exports contain CPU data only.
    Settings and n/d/p stay fixed; use load() to replace data. A workspace is
    used serially on its original CUDA device/stream. allocate() starts empty;
    the original constructor still accepts and immediately loads X/y.
    Affine/degree-1/degree-2 polynomial features are constructed once on GPU
    from uploaded X, then shared with the least-squares workspace without copying.
    Omitted initial states are generated on GPU: Gaussian unit w and median(Xw)
    for z. One CPU-derived seed initializes the workspace's advancing cuRAND RNG.
    """

    def __init__(self, splitter, X, y, *, temperature=None, reorder=True):
        temp = splitter.temperature if temperature is None else temperature
        X, y, temp = _batch_data(X, y, temp)
        self._allocate(splitter, *X.shape, reorder=reorder)
        self._load(X, y, temp)

    @classmethod
    def allocate(cls, splitter, capacity, n, d, *, reorder=True, stream=None):
        """Reserve GPU and CPU packing/control buffers for an entire build."""
        ws = cls.__new__(cls)
        ws._allocate(splitter, capacity, n, d, reorder=reorder, stream=stream)
        return ws

    def _allocate(self, splitter, capacity, n, d, *, reorder, stream=None):
        self.K, self.n, self.d = _dimensions(capacity, n, d)
        _validate_settings(splitter, self.d, reorder)
        self.s, self.capacity, self.k = splitter, self.K, 0
        self.reorder = bool(reorder)
        self.seed = int(splitter._rng().integers(0, np.iinfo(np.int64).max))
        self.M = (self.n + 255) // 256
        self.S = splitter.max_backtracks + 1
        self.H = min(splitter.backtrack_batch_size, self.S)
        self.required = splitter._required_side_points(self.n)
        self.states = [None] * self.K
        model = splitter.model_template
        affine = type(model) is AffineRidgeModel
        if not affine:
            from sklearn.preprocessing import PolynomialFeatures

            # fit() records dimension/basis metadata only; no CPU transform.
            # The fixed basis is identical for every node and exported model.
            state = PolynomialFeatures(degree=model.degree, include_bias=model.include_bias)
            state.fit(np.zeros((1, self.d)))
            self.states = [state] * self.K

        import cupy as cp

        self.cp = cp
        self.stream = cp.cuda.get_current_stream() if stream is None else stream
        self.blas_handle = None
        self.rng_handle = None
        from cupy_backends.cuda.libs import cublas, curand

        self.blas, self.curand = cublas, curand
        with self.stream:
            self.features = FeaturePlan(self.d, degree=1 if affine else model.degree,
                                        include_bias=True if affine else model.include_bias,
                                        bias_last=affine)
            self.p = self.features.p
            self.x = cp.empty((self.K, self.n, self.d), dtype=cp.float64)
            self.ls = BatchedCholeskyWorkspace.allocate(
                self.K, self.n, self.p, model.ridge, splitter.weight_floor, stream=self.stream)
            self.blas_handle = self.ls.blas_handle  # One BLAS context for all products.
            self.rng_handle = curand.createGenerator(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10)
            curand.setPseudoRandomGeneratorSeed(self.rng_handle, self.seed)
            curand.setStream(self.rng_handle, self.stream.ptr)
            # cuRAND's normal generator writes pairs, including for odd count*d.
            self._normal = cp.empty(2 * ((self.K * self.d + 1) // 2), dtype=cp.float64)
            self.functions = {**kernels(cp.cuda.Device().id), **_bookkeeping(cp.cuda.Device().id)}
            self.temp = cp.ones(self.K)
            self.theta, self.last, self.best_theta = [cp.empty((self.K, self.d + 1)) for _ in range(3)]
            self.best_beta = cp.empty_like(self.ls._rhs)
            self.u, self.v, self.pi, self.coeff = [cp.zeros((self.K, self.n)) for _ in range(4)]
            self.pred, self.errors = [cp.empty((self.K, self.n, 2)) for _ in range(2)]
            self.gw, self.pw = cp.empty((self.K, self.d)), cp.empty_like(self.theta)
            self.partials, self.totals = cp.empty((self.K, 3, self.M)), cp.empty((self.K, 3))
            self.geometry, self.packet = cp.empty((self.K, 4)), cp.zeros((self.K, 7))
            self.active, self.searching, self.selected, self.ids = [cp.empty(self.K, dtype=cp.int32) for _ in range(4)]
            self.alpha, self.inv_r = cp.zeros(self.K), cp.ones(self.K)
            self.alphas = cp.empty((self.K, self.S))
            self.trial_partials, self.losses = cp.empty((self.K, self.H, self.M)), cp.empty((self.K, self.H))
            self.bt = cp.empty((4, self.K))
            self.value, self.step = self.bt[0], self.bt[1]
            self.products = {
                "u": self._product(self.x, self.theta[:, :-1, None], self.u[:, :, None]),
                "v": self._product(self.x, self.pw[:, :-1, None], self.v[:, :, None]),
                "pred": self._product(self.ls.phi, self.ls.coefficients.swapaxes(-1, -2), self.pred),
                "gw": self._product(self.x.swapaxes(-1, -2), self.coeff[:, :, None], self.gw[:, :, None]),
            }
        self.all_ids = np.arange(self.K, dtype=np.int32)
        self.order = self.all_ids.copy()
        self.run_ids = self.all_ids[:0]
        self.run_mask = np.zeros(self.K, dtype=bool)
        self.mask = None
        self.fit_sizes = []
        self.host_X = np.empty((self.K, self.n, self.d))
        self.host_y = np.empty((self.K, self.n))
        self.X, self.y = self.host_X[:0], self.host_y[:0]
        self.temperature = np.ones(self.K)
        self._mask_host = np.empty(self.K, dtype=np.int32)
        self._active_host = np.empty(self.K, dtype=bool)
        self._alpha_host = np.empty(self.K)
        self._theta_init = np.zeros((self.K, self.d + 1))
        self._packet_host = np.zeros((self.K, 7))
        self._hard_host = np.zeros((self.K, 7))
        self._search_host = np.empty((4, self.K))
        self._theta_host = np.empty((self.K, self.d + 1))
        self._beta_host = np.empty((2, self.K, self.p))

    def _check_open(self):
        if self.blas_handle is None:
            raise RuntimeError("The workspace is closed.")
        if self.cp.cuda.Device().id != self.ls.device_id:
            raise ValueError("Use the workspace on the device where it was created.")

    def load(self, X, y, *, temperature=None):
        """Load one packed CPU cohort into fixed GPU buffers; overwrite Phi.

        Shapes are (k,n,d)/(k,n), with 0<k<=capacity. Builders may pack directly
        into host_X/host_y and pass their occupied prefixes to avoid a second
        CPU copy. Earlier exported results own their data and remain valid.
        """
        self._check_open()
        temp = self.s.temperature if temperature is None else temperature
        X, y, temp = _batch_data(X, y, temp)
        if X.shape[0] > self.K or X.shape[1:] != (self.n, self.d):
            raise ValueError("X/y must have shapes (k,n,d)/(k,n), 0<k<=capacity.")
        return self._load(X, y, temp)

    def _load(self, X, y, temp):
        k = X.shape[0]
        self.k, self.ls.k = 0, 0
        for dst, src in ((self.host_X[:k], X), (self.host_y[:k], y)):
            if dst.ctypes.data != src.ctypes.data or dst.strides != src.strides:
                np.copyto(dst, src)
        self.temperature[:k] = temp
        with self.stream:
            self.x[:k].set(self.host_X[:k], stream=self.stream)
            self.temp[:k].set(self.temperature[:k], stream=self.stream)
            self.features.build(self.x[:k], out=self.ls.phi[:k])
            self.ls.load(self.ls.phi[:k], self.host_y[:k])
            self.active.fill(0)
            self.searching.fill(0)
            self.selected.fill(-1)
            self.inv_r.fill(1.)
            self.packet.fill(0.)
        self.k = k
        self.X, self.y = self.host_X[:k], self.host_y[:k]
        self.order[:] = self.all_ids
        self.run_ids = self.all_ids[:k]
        self.run_mask.fill(False)
        self.run_mask[:k] = True
        self.mask = None
        self._packet_host.fill(0.)
        self._hard_host.fill(0.)
        self.fit_sizes.clear()
        return self

    def _product(self, a, b, out):
        return BatchedMatmul(self.cp, a, b, out, self.blas_handle, self.stream)

    def call(self, name, grid, *args):
        args = tuple(np.int32(a) if isinstance(a, (int, np.integer, bool)) else
                     np.float64(a) if isinstance(a, (float, np.floating)) else a for a in args)
        self.functions[name](grid, (256,), args)

    def _set_active(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if self.mask is not None and np.array_equal(mask, self.mask):
            return
        np.copyto(self._active_host, mask)
        self.mask = self._active_host
        self.count = _partition(self.order, mask)
        self._mask_host[:] = mask
        self.active.set(self._mask_host, stream=self.stream)
        self.ids[:self.count].set(self.order[:self.count], stream=self.stream)
        for plan in self.products.values():
            plan.select(self.ids, self.count)

    def _fit(self, *, hard=False, phase=None):
        # Same GEMM/solver path in both modes; only the submitted prefix differs.
        nodes = self.order[:self.count] if self.reorder else self.run_ids
        self.ls.fit(self.pi, nodes=nodes, weight_floor=0. if hard else None)
        self.fit_sizes.append((phase or ("hard" if hard else "soft"), self.ls.last_batch_size // 2))
        self.products["pred"].run()

    def start(self, theta, mask):
        with self.stream:
            self.pi[:self.k].fill(.5)
            self._set_active(mask)
            if theta is None:
                self._random_theta()
            else:
                self.theta.set(theta, stream=self.stream)
                self.products["u"].run()
            self._gates(advance=False)

    def _random_theta(self):
        """Initialize selected nodes and retain u=Xw for the first gate call."""
        count = self.count
        if not count:
            return
        draws = 2 * ((count * self.d + 1) // 2)
        self.curand.generateNormalDouble(self.rng_handle, self._normal.data.ptr, draws, 0., 1.)
        self.call("initial_directions", (count,), self.d, self.ids, self._normal, self.theta)
        self.products["u"].run()
        # v (normally Xp_w) and alpha are free until the first GD/search step.
        self.call("initial_scores", (self.M, count), self.n, self.ids, self.u, self.v)
        self.cp.median(self.v[:count], axis=1, out=self.alpha[:count], overwrite_input=True)
        self.call("initial_offsets", ((count + 255) // 256,), count, self.d,
                  self.ids, self.alpha, self.theta)

    def _gates(self, *, advance=False, hard=False):
        self.call("batched_gates", (self.M, self.k), self.n, self.d, advance, hard,
                  self.active, self.temp, self.theta, self.step, self.v, self.inv_r, self.u, self.pi)

    def current(self, mask):
        """Fit and evaluate; download one (K,7) control array."""
        with self.stream:
            self._set_active(mask)
            self._fit()
            self.call("batched_residual_stats", (self.M, self.k), self.n, self.active,
                      self.temp, self.ls.y, self.pred, self.pi, self.errors, self.coeff, self.partials)
            self.products["gw"].run()
            self.cp.sum(self.partials[:self.k], axis=2, out=self.totals[:self.k])
            self.call("batched_direction", (self.k,), self.K, self.d, self.active,
                      self.theta, self.gw, self.totals, self.ls._ridge, self.ls._status,
                      self.pw, self.geometry, self.packet)
            return self.packet.get(stream=self.stream, out=self._packet_host)

    def _save_feasible(self, packet, mask, it, feasible):
        good = mask & (packet[:, 6] == 0) & (packet[:, 3] >= self.required) & (self.n - packet[:, 3] >= self.required)
        feasible[good] = it[good]
        self.call("save_feasible", ((self.k * (self.d + 1) + 255) // 256,), self.K,
                  self.n, self.d, self.required, self.active, self.theta, self.packet, self.last)

    def search(self, mask, controllers):
        """Search/retry independently, retaining accepted nodes' device results."""
        s = self.s
        retry, accepted = mask.copy(), np.zeros(self.K, dtype=bool)
        failures = np.zeros(self.K, dtype=int)
        physical, logical = np.zeros(self.K, dtype=int), np.zeros(self.K, dtype=int)
        reasons = [None] * self.K
        self.selected.fill(-1)
        self.value.fill(np.inf)
        self.step.fill(0.)
        state = self._search_host
        while retry.any():
            self._mask_host[:] = retry
            self.searching.set(self._mask_host, stream=self.stream)
            self._alpha_host[:] = [c.alpha if s.adaptive_alpha else s.alpha0 for c in controllers]
            self.alpha.set(self._alpha_host, stream=self.stream)
            self.call("alpha_steps", ((self.k + 255) // 256,), self.K, self.S, s.rho,
                      self.searching, self.alpha, self.alphas)
            pending = retry.copy()
            for offset in range(0, self.S, self.H):
                self.call("batched_trial_losses", (self.M, self.H, self.k), self.n, self.d,
                          self.S, offset, self.searching, self.temp, self.theta, self.u, self.v,
                          self.errors, self.geometry, self.alphas, self.trial_partials)
                self.call("batched_trial_reduce", (self.H, self.k), self.M, self.searching,
                          self.trial_partials, self.losses)
                self.call("batched_select_trial", (self.k,), self.S, self.H, offset, s.armijo_c,
                          self.losses, self.alphas, self.packet, self.searching, self.selected,
                          self.value, self.step)
                self.call("search_state", ((self.k + 255) // 256,), self.K, self.selected, self.searching, self.bt)
                state = self.bt.get(stream=self.stream, out=self._search_host)
                count = min(self.H, self.S - offset)
                physical[pending] += count
                logical[pending] += np.where(state[2, pending] >= 0,
                                             state[2, pending].astype(int) - offset + 1, count)
                pending = state[3].astype(bool)
                if not pending.any():
                    break
            again = np.zeros(self.K, dtype=bool)
            for j in np.flatnonzero(retry):
                ok = state[2, j] >= 0
                result = LineSearchResult(np.empty(0), float(state[0, j]), float(state[1, j]),
                                          bool(ok), int(state[2, j]) if ok else self.S - 1)
                if s.adaptive_alpha:
                    controllers[j].update(result, s.rho)
                if ok:
                    accepted[j] = True
                else:
                    failures[j] += 1
                    if not s.adaptive_alpha:
                        reasons[j] = "line_search_failed"
                    elif failures[j] >= s.max_line_search_failures:
                        reasons[j] = "max_line_search_failures"
                    elif controllers[j].alpha <= s.alpha_min:
                        reasons[j] = "alpha_min_reached"
                    else:
                        again[j] = True
            retry = again
        return accepted, state, physical, logical, reasons

    def advance(self, accepted, it):
        self.call("batched_commit_theta", (self.k,), self.d, self.selected, self.step,
                  self.pw, self.geometry, self.theta, self.inv_r)
        refresh = accepted & (it % self.s.Xw_refresh == 0) if self.s.Xw_refresh else np.zeros(self.K, bool)
        if refresh.any():
            self._set_active(refresh)
            self.products["u"].run()
            self._gates()
        cached = accepted & ~refresh
        if cached.any():
            self._set_active(cached)
            self._gates(advance=True)

    def _parent_loss(self):
        self._set_active(self.run_mask)
        self.pi.fill(.5)
        self._fit(hard=True, phase="parent")
        # The common factor 1/2 cancels in G, h and the trace-scaled ridge.
        self.call("hard_stats", (self.k,), self.K, self.n, self.active, self.ls.y,
                  self.pred, self.pi, self.ls._ridge, self.ls._status, self.packet)
        packet = self.packet.get(stream=self.stream, out=self._hard_host)
        return packet[:, 0].copy(), (packet[:, 6] == 0) & self.run_mask

    def _finish(self, packet, valid, feasible, hard_refit):
        fallback = valid & (np.minimum(packet[:, 3], self.n - packet[:, 3]) < self.required)
        fallback &= self._fallback & (feasible >= 0)
        self._set_active(fallback)
        self.call("copy_theta", ((self.k * (self.d + 1) + 255) // 256,), self.K, self.d,
                  self.active, self.last, self.theta)
        self._set_active(valid)
        self.call("normalize_theta", (self.k,), self.d, self.active, self.theta)
        self.products["u"].run()
        self._gates(hard=True)
        if valid.any():
            if hard_refit:
                self._fit(hard=True)
                self.call("hard_stats", (self.k,), self.K, self.n, self.active, self.ls.y,
                          self.pred, self.pi, self.ls._ridge, self.ls._status, self.packet)
            else:
                self.call("partition_counts", (self.k,), self.n, self.active, self.pi, self.packet)
        return self.packet.get(stream=self.stream, out=self._hard_host), fallback

    def _run(self, theta, parent_ok, hard_refit=True):
        s, K = self.s, self.K
        active = parent_ok.copy()
        it, feasible = np.zeros(K, dtype=int), np.full(K, -1, dtype=int)
        converged, reason = np.zeros(K, bool), ["max_iters"] * K
        errors = [None] * K
        for j in self.run_ids:
            if not parent_ok[j]:
                errors[j] = SplitNotFoundError("cholesky_failure", "Parent Cholesky fit failed.")
        controllers = [AdaptiveAlpha(s.alpha0, s.alpha_min, s.alpha_max, s.alpha_grow,
                                      s.alpha_recovery, s.heavy_backtrack_threshold) for _ in range(K)]
        self.start(theta, active)
        packet = self.current(active) if active.any() else np.zeros((K, 7))
        tolerance = s.grad_atol + s.grad_rtol * packet[:, 1]
        history = [{"soft_loss_history": [float(row[0])], "projected_grad_norm_history": [float(row[1])],
                    "step_size_history": [], "backtrack_history": [], "line_search_loss_history": []} for row in packet]
        physical, logical = np.zeros(K, int), np.zeros(K, int)
        self._save_feasible(packet, active, it, feasible)
        while active.any():
            for j in np.flatnonzero(active):
                if packet[j, 6]:
                    errors[j] = SplitNotFoundError("cholesky_failure", "GPU fit or gradient failed.")
                    active[j] = False
                elif it[j] >= s.max_iters:
                    active[j] = False
                elif packet[j, 1] <= tolerance[j]:
                    converged[j], reason[j], active[j] = True, "gradient_tolerance", False
                elif packet[j, 2] >= 0:
                    reason[j], active[j] = "non_descent_direction", False
            if not active.any():
                break
            self._set_active(active)
            self.products["v"].run()
            accepted, state, actual, examined, stopped = self.search(active, controllers)
            physical += actual
            logical += examined
            for j in np.flatnonzero(active & ~accepted):
                reason[j] = stopped[j]
            active &= accepted
            if not active.any():
                break
            it[active] += 1
            self.advance(active, it)
            packet = self.current(active)
            self._save_feasible(packet, active, it, feasible)
            for j in np.flatnonzero(active):
                h = history[j]
                h["soft_loss_history"].append(float(packet[j, 0]))
                h["projected_grad_norm_history"].append(float(packet[j, 1]))
                h["step_size_history"].append(float(state[1, j]))
                h["backtrack_history"].append(int(state[2, j]))
                h["line_search_loss_history"].append(float(state[0, j]))
        valid = self.run_mask & np.array([error is None for error in errors])
        hard, fallback = self._finish(packet, valid, feasible, hard_refit)
        metadata = []
        for j in range(K):
            if valid[j] and not hard_refit and hard[j, 6]:
                errors[j] = SplitNotFoundError("nonfinite_split", "Final partition is nonfinite.")
            elif valid[j] and min(hard[j, 3], self.n - hard[j, 3]) < self.required:
                errors[j] = SplitNotFoundError("min_side_points", "Hard split has too few samples on one side.")
            elif valid[j] and hard[j, 6]:
                errors[j] = SplitNotFoundError("cholesky_failure", "Hard Cholesky fit failed.")
            meta = dict(history[j], optimizer="cupy_batched_cholesky", reorder=self.reorder,
                        initialization="gpu" if theta is None else "provided",
                        initialization_seed=self.seed if theta is None else None,
                        gpu_solver="cholesky", solver_fallback=False, condition_estimation=False,
                        complementary_grams=False, backtrack_batch_size=self.H, Xw_refresh=s.Xw_refresh,
                        temperature=float(self.temperature[j]), n_iters=int(it[j]), stop_reason=reason[j],
                        converged=bool(converged[j]), grad_tolerance=float(tolerance[j]),
                        initial_grad_norm=history[j]["projected_grad_norm_history"][0],
                        final_grad_norm=history[j]["projected_grad_norm_history"][-1],
                        adaptive_alpha=s.adaptive_alpha, refit_during_line_search=False,
                        physical_line_search_candidates=int(physical[j]), logical_line_search_candidates=int(logical[j]),
                        speculative_line_search_candidates=int(physical[j] - logical[j]),
                        final_alpha=controllers[j].alpha if s.adaptive_alpha else s.alpha0,
                        hard_refit=hard_refit, n_left=self.n - int(hard[j, 3]), n_right=int(hard[j, 3]),
                        left_loss=float(hard[j, 1]) if hard_refit else None,
                        right_loss=float(hard[j, 2]) if hard_refit else None,
                        required_side_points=self.required)
            meta["final_soft_metadata"] = {
                "temperature": float(self.temperature[j]), "model_type": type(s.model_template).__name__,
                "models_refit": True, "prepared_design": self.states[j] is not None,
                **{f"{side}_solve": {"solver_requested": "normal", "solver_used": "cholesky",
                                     "condition_estimator": None, "cond_estimate": None, "rank": None,
                                     "fallback_reason": None, "ridge_effective": float(packet[j, 4 + k])}
                   for k, side in enumerate(("left", "right"))},
            }
            if fallback[j]:
                meta.update(min_side_fallback=True, fallback_iteration=int(feasible[j]))
            if errors[j] is not None:
                meta["stop_reason"] = errors[j].reason
                errors[j].diagnostics = meta
            metadata.append(meta)
        return hard, fallback, metadata, errors

    def optimize(self, theta0=None, *, parent_loss=None, nodes=None, min_side_fallback=None, hard_refit=True):
        """Run restarts, retain best candidates on GPU, and export once at the end.

        nodes selects occupied slot IDs, in return order; default is all k.
        For r selected nodes, theta0 is (r,d+1) or (R,r,d+1), and parent_loss
        is (r,). Omitted starts are generated entirely on GPU; the workspace
        RNG advances across retries/restarts/loads. Selected retries reuse data/features. min_side_fallback
        overrides the policy for this call, e.g. only on the final attempt.
        hard_refit=False (one restart only) returns geometry/counts with no hard
        models or loss/gain measurements, for builders that fit topped-up children.
        """
        self._check_open()
        if not self.k:
            raise RuntimeError("Load data before optimizing.")
        ids = self.all_ids[:self.k] if nodes is None else _node_ids(nodes, self.k)
        count = len(ids)
        R = max(1, int(self.s.n_restarts))
        if not isinstance(hard_refit, (bool, np.bool_)):
            raise ValueError("hard_refit must be a boolean.")
        if not hard_refit and self.s.n_restarts != 1:
            raise ValueError("Skipping hard refits requires n_restarts == 1.")
        initial = None if theta0 is None else np.asarray(theta0, dtype=float)
        if initial is not None and (initial.shape not in ((count, self.d + 1), (R, count, self.d + 1))
                                    or not np.isfinite(initial).all()):
            raise ValueError("theta0 must be finite with shape (r,d+1) or (R,r,d+1).")
        if parent_loss is not None:
            parent_loss = np.asarray(parent_loss, dtype=float)
            if parent_loss.shape != (count,) or not np.isfinite(parent_loss).all() or np.any(parent_loss < 0):
                raise ValueError("parent_loss must be finite and nonnegative with shape (r,).")
        if min_side_fallback is not None and not isinstance(min_side_fallback, (bool, np.bool_)):
            raise ValueError("min_side_fallback must be a boolean.")
        if not count:
            return []
        self.run_ids = ids
        self.run_mask.fill(False)
        self.run_mask[ids] = True
        self._fallback = self.s.min_side_fallback if min_side_fallback is None else bool(min_side_fallback)
        best_loss, best_fallback = np.full(self.K, np.inf), np.ones(self.K, bool)
        best_meta, best_hard = [None] * self.K, np.empty((self.K, 7))
        failures = [None] * self.K
        self.fit_sizes.clear()
        self.ls.fit_count = 0
        with self.stream:
            if parent_loss is None:
                parent_loss, parent_ok = self._parent_loss()
            else:
                supplied = parent_loss
                parent_loss = np.zeros(self.K)
                parent_loss[ids] = supplied
                parent_ok = self.run_mask.copy()
            for r in range(R):
                theta = None
                if initial is not None and (r == 0 or initial.ndim == 3):
                    theta = self._theta_init
                    theta.fill(0.)
                    starts = initial if initial.ndim == 2 else initial[r]
                    for j, row in zip(ids, starts):
                        theta[j] = project_unit_w(row)
                hard, fallback, metadata, errors = self._run(theta, parent_ok, hard_refit)
                if not hard_refit:
                    # No restart ranking, coefficient copy, or model export is needed.
                    theta = self.theta.get(stream=self.stream, out=self._theta_host)
                    return [_partition_result(theta[j], self.n, hard[j, 3], parent_loss[j],
                                              metadata[j]["converged"], metadata[j])
                            if errors[j] is None else errors[j] for j in ids]
                keep = np.zeros(self.K, bool)
                for j in ids:
                    if errors[j] is not None:
                        failures[j] = errors[j]
                    elif best_meta[j] is None or (best_fallback[j] and not fallback[j]) or (
                            best_fallback[j] == fallback[j] and hard[j, 0] < best_loss[j]):
                        keep[j] = True
                        best_meta[j], best_hard[j], best_loss[j], best_fallback[j] = metadata[j], hard[j], hard[j, 0], fallback[j]
                self._set_active(keep)
                self.call("keep_best", ((self.k * (self.d + 1 + 2 * self.p) + 255) // 256,),
                          self.K, self.d, self.p, self.active, self.theta, self.ls._rhs,
                          self.best_theta, self.best_beta)
            theta = self.best_theta.get(stream=self.stream, out=self._theta_host)
            beta = self.best_beta.get(stream=self.stream, out=self._beta_host).transpose(1, 0, 2)
        return [self._export(j, theta[j], beta[j], best_hard[j], best_meta[j], parent_loss[j])
                if best_meta[j] is not None else failures[j] for j in ids]

    def _export(self, j, theta, beta, hard, meta, parent_loss):
        models = []
        for side in range(2):
            model = self.s.model_template.clone()
            model.coef_ = beta[side].copy()
            if self.states[j] is not None:
                model.transformer_ = self.states[j]
            model.solve_result_ = RidgeSolveResult(
                coefficients=model.coef_, solver_requested="normal", solver_used="cholesky",
                condition_estimator=None, cond_estimate=None, ridge_effective=float(hard[4 + side]), rank=None)
            models.append(model)
        right = int(hard[3])
        gain = float(parent_loss - hard[0])
        return SplitResult(w=theta[:-1].copy(), z=float(theta[-1]), left_model=models[0], right_model=models[1],
                           loss=float(hard[0]), parent_loss=float(parent_loss), split_gain=gain,
                           relative_split_gain=gain / max(float(parent_loss), 1e-12),
                           n_left=self.n - right, n_right=right, converged=meta["converged"],
                           n_iters=meta["n_iters"], stop_reason=meta["stop_reason"], metadata=meta)

    def close(self):
        if getattr(self, "blas_handle", None) is not None:
            with self.cp.cuda.Device(self.ls.device_id):
                self.ls.close()  # Owns the shared BLAS and solver handles.
                self.blas_handle = None
                if self.rng_handle is not None:
                    self.curand.destroyGenerator(self.rng_handle)
                    self.rng_handle = None
                self.products.clear()
                self.features = None
                for name, value in list(vars(self).items()):
                    if isinstance(value, (self.cp.ndarray, np.ndarray)):
                        setattr(self, name, None)
                self.k = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def optimize_batch(splitter, X, y, *, theta0=None, parent_loss=None, temperature=None, reorder=True,
                    hard_refit=True):
    """Convenience entry point; use the workspace directly to reuse its buffers."""
    with BatchedOptimizerWorkspace(splitter, X, y, temperature=temperature, reorder=reorder) as ws:
        return ws.optimize(theta0, parent_loss=parent_loss, hard_refit=hard_refit)
