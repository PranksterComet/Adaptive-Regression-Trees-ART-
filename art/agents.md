# Adaptive Regression Trees (ART): Project Context

Last implementation review: 2026-09-01

This document is a technical handoff for agents and collaborators working on
ART. It describes the intended research problem, the mathematical ideas behind
the current implementation, what is actually implemented, known weaknesses,
and the work needed for a defensible paper and a useful library.

The repository is an active research prototype. Treat statements labeled
**implemented** as descriptions of the current code, **experimental** as code
that exists but is not fully integrated or validated, and **planned** as design
intent rather than current behavior.

## 1. Vision

ART is an oracle-driven regression tree for approximating an expensive scalar
function over a bounded continuous domain. The central goal is to spend oracle
queries adaptively: fit a simple local model where one works, and subdivide only
where the approximation error remains too large.

Unlike an axis-aligned CART tree, an ART node uses an oblique hyperplane

```text
H(w, z) = {x : w^T x >= z},    ||w||_2 = 1,
```

and fits a separate regression model on each side. The intended system should
support interchangeable:

- domains and region representations;
- samplers for the current region;
- local regression model classes;
- split objectives and split optimizers;
- error metrics and stopping rules;
- sample-budget and adaptive-sampling policies.

The long-term use case is surrogate construction for deterministic scientific
or engineering oracles, especially when the function is globally complicated
but locally well approximated by low-degree models. The main resource metric is
the number of oracle evaluations, with wall-clock time and memory as secondary
constraints.

## 2. Population Problem

Let `D` be a bounded domain with target probability measure `nu`; the current
experiments normally use the uniform measure. Let `f : D -> R` be the oracle and
let `M` be a local model class. At a node corresponding to region `R subset D`,
the ideal split solves

```text
min_{w,z} [
    P_R(A)     inf_{phi_L in M} E_R[(f(X)-phi_L(X))^2 | X in A]
  + P_R(A^c)   inf_{phi_R in M} E_R[(f(X)-phi_R(X))^2 | X in A^c]
],

A = {x in R : w^T x < z}.
```

Equivalently, this is the sum of the two unnormalized error integrals over the
child regions. ART replaces those expectations by samples from the current
polytope and solves a smoothed, nonconvex empirical optimization problem.

The recursive objective is not currently posed as one globally optimized tree
problem. Construction is greedy: each node accepts the locally best split found
by its splitter, subject to balance and empirical MSE-gain checks.

## 3. Design Principles

The current architecture follows these principles:

1. **Sample once, reuse locally.** A node's raw coordinates and oracle values are
   reused for its parent fit, split optimization, hard split fit, retries, and
   temperature candidates.
2. **Prepare expensive features once.** Polynomial features are transformed once
   per node and shared by all weighted refits at that node.
3. **Keep routing geometry in raw coordinates.** Hyperplanes always act on the
   original `d`-dimensional `X`, even when leaf models use a larger feature basis.
4. **Separate numerical optimization from tree policy.** A splitter proposes a
   hard partition and models; the builder decides whether its empirical gain is
   sufficient and what to do after failure.
5. **Preserve diagnostics when requested.** Failed splits should remain
   inspectable rather than disappearing behind a generic leaf status.
6. **Count oracle queries explicitly.** Geometry-only pilot chains and model
   refits do not increment the oracle-query count.

## 4. Module Map

| Module | Responsibility | Status |
| --- | --- | --- |
| `domain.py` | Box domains, polytope regions `A x <= b`, and hyperplane subdivision | Implemented |
| `sampling.py` | Exact box sampling, hit-and-run, covariance-shaped directions, ACF diagnostics | Implemented; adaptive thinning not integrated |
| `models.py` | Model protocols, affine/polynomial ridge, kernel ridge, prepared features | Partly generic |
| `metrics.py` | MSE, relative L2, pointwise relative errors and quantiles | Implemented |
| `objectives.py` | Cached soft oblique objective and gradient | Implemented with mathematical caveats |
| `optimizers.py` | Armijo backtracking and adaptive initial step controller | Implemented |
| `splitters.py` | Soft oblique and hinge-affine splitters, split diagnostics | Implemented |
| `temperature.py` | Geometric temperature scales and candidate grids | Implemented |
| `presets.py` | Shared benchmark defaults and effective-dimension side-size policy | Implemented |
| `builder.py` | Sampling, fitting, stopping, splitting, retries, recursion, query accounting | Implemented basic builder |
| `tree.py` | Node structures, batched prediction, traversal, joblib persistence | Implemented |
| `timing.py` | Optional aggregate build-time profiling | Implemented; intentionally coarse |

The `examples/` directory contains benchmark functions, splitter stress tests,
tree visualizations, and diagnostic notebooks. The `tests/` directory contains
unit tests plus statistical diagnostic scripts; not every diagnostic script is
a deterministic unit test.

## 5. Current Tree-Building Algorithm

For a region `R`, node sample target `n_node`, model template `M`, and stopping
metric `e`, the builder performs:

```text
BUILD(R, inherited samples, depth):
    top up inherited samples to n_node using the region sampler
    query f only at newly sampled points
    prepare the model design matrix once, if supported
    fit a fresh model to all node samples
    compute training error e(y, model(X)) and training MSE

    if error <= tolerance:
        return leaf(status="tolerance_met")
    if depth == max_depth:
        return leaf(status="max_depth")

    choose or tune soft-split temperature, if applicable
    run splitter on the same node samples
    optionally retry selected failures with a new initialization

    if no valid split is found:
        return diagnostic leaf
    if hard-split MSE gain is too small:
        return diagnostic leaf

    create left and right child polytopes
    pass each child the parent samples already on its side
    recursively build both children
```

### 5.1 Sample budget

By default,

```text
n_node = sample_multiplier * p_eff,
```

where `p_eff` is the number of fitted coefficients. It is `d+1` for affine
models and

```text
p_eff = binomial(d + m, m)
```

for degree-`m` polynomial features with a bias. The default multiplier is 50.
An explicit integer or callable sample-count policy can replace this rule.

This is a heuristic, not a proved sample-complexity result. Polynomial feature
counts grow combinatorially, making high-dimensional/high-degree fits expensive.

The shared 2D/high-dimensional benchmark preset currently uses affine leaves,
maximum depth 10, one splitter restart, three conditional failure retries,
unregularized fits (`ridge=0`) with the `auto` solver, fixed local temperature
scaling with `c=1`, thinning 20, covariance-shaped sampling, stored diagnostics,
and no stored node samples. When `min_side_points` is not given, the benchmark
harness resolves it to `p_eff`. These are experiment-harness defaults, not
universal guarantees of the lower-level builder API.

### 5.2 Root and child sampling

- An axis-aligned `BoxDomain` is sampled exactly and independently at the root by
  default.
- A non-box root and child regions use hit-and-run.
- A child inherits parent samples already on its side. If there are too many,
  they are randomly subsampled; if too few, the child is topped up.
- A random inherited point seeds the new hit-and-run chain.
- Child burn-in is normally zero under the intended argument that a point drawn
  uniformly from the parent, conditioned on entering the child, is uniform in
  the child. This argument is exact only under ideal parent sampling and does not
  make the combined finite sample independent.

The inherited sample values are not re-queried. `oracle_queries` counts only
points evaluated during tree construction; test-set and plotting evaluations in
examples are separate.

### 5.3 Node stopping and split acceptance

The node stopping metric is user-selectable. Current metrics include MSE,
relative L2 error, and median/max pointwise relative error. It is evaluated on
the same samples used to fit the node model.

At present, the same configured metric value is used for stopping and stored as
`fit_error`. Relative metrics use one denominator floor in both places. A
separate absolute-or-relative stopping rule with an uncapped/raw diagnostic
relative error has been discussed but is **not implemented**. This distinction
matters for functions that approach or cross zero.

Split acceptance is separate and always based on hard-split training MSE:

```text
gain_abs = MSE_parent - MSE_split
gain_rel = gain_abs / max(MSE_parent, 1e-12).
```

The builder requires both values to exceed their configured thresholds. With
the default absolute threshold zero, a split must still have strictly positive
training MSE gain.

The distinction is intentional: a stopping metric describes approximation
quality, while MSE gain provides an additive expectation-based split criterion.
It also means tuning and acceptance may not directly optimize the user's chosen
stopping metric.

## 6. Regression Models and Prepared Features

### 6.1 Interfaces

`RegressionModel` requires `fit`, `predict`, and `clone`. The soft splitter
additionally requires `WeightedRegressionModel.fit_weighted`. Models with an
expensive explicit basis may implement `PreparedFeatureModel`, which adds:

- `prepare_design(X)`;
- unweighted and weighted fits from a prepared design;
- prediction from a prepared design.

The raw `X` remains available for hyperplane routing. A `PreparedDesign` stores
the transformed matrix, input dimension, feature signature, and fitted transform
state. Subsets preserve row alignment without recomputing features.

### 6.2 Weighted trace-scaled ridge

Affine and polynomial models solve

```text
G       = Phi^T W Phi,
lambda  = ridge * trace(G) / p,
beta    = (G + lambda I)^{-1} Phi^T W y,
```

where `Phi` is the design matrix and `p` its number of columns. This makes the
ridge parameter scale with the mean eigenvalue of the weighted Gram matrix.
Weights are floored before forming `G`, currently at `1e-12` by default.

The model API supports `auto`, `normal`, `qr`, and `svd` solver modes. Normal
mode uses a Cholesky factorization of the regularized Gram matrix. Auto mode
estimates its reciprocal 1-norm condition with LAPACK `dpocon` and uses the
Cholesky solution only when `rcond >= 1e-10`; otherwise it solves the equivalent
augmented ridge system with pivoted QR. Explicit QR and SVD use SciPy's `gelsy`
and `gelsd` LAPACK drivers. Solver results include the requested and resolved
solver, effective ridge, rank when available, a condition estimate and its
estimator, and any fallback reason. Raw feature scaling remains unresolved.

### 6.3 Actual model support

- `AffineRidgeModel`: fully usable in the builder, soft splitter, and HRT.
- `PolynomialRidgeModel`: fully usable in the builder and soft splitter; feature
  expansion is reused once per node.
- `KernelRidgeModel`: usable as a standalone fit/predict model, but it currently
  lacks weighted fitting and an effective finite sample-budget dimension. It is
  therefore **not currently compatible with the generic soft splitter/builder
  path**. The library should not claim otherwise until this contract is added or
  a different inner optimizer is introduced.

## 7. Soft Oblique Splitter

### 7.1 Sigmoid relaxation

For `theta = (w, z)` with `||w||_2 = 1`, define the right-side soft membership

```text
pi_i(theta) = sigmoid((w^T x_i - z) / T).
```

`T > 0` is the temperature. Small `T` approximates a hard split; large `T`
blends both models over a wider slab around the hyperplane.

At a given `theta`, fit models with weights

```text
q_L,i = 1 - pi_i,       q_R,i = pi_i.
```

The empirical soft loss reported by the objective is

```text
J_T(theta) = (1/n) sum_i [
    (1-pi_i) (y_i-phi_L(x_i))^2
  + pi_i     (y_i-phi_R(x_i))^2
].
```

This is a variable-projection idea: regression coefficients are inner variables
solved by weighted regression, while `(w,z)` are outer variables optimized by
projected gradient descent.

### 7.2 Gradient used by the code

Let

```text
r_L,i = y_i - phi_L(x_i),
r_R,i = y_i - phi_R(x_i),
a_i   = (r_R,i^2-r_L,i^2) pi_i(1-pi_i) / n.
```

The implemented gradient is

```text
grad_w J = X^T a / T,
grad_z J = -sum_i a_i / T.
```

For exact unregularized weighted least-squares inner minimizers, this follows
from the envelope theorem: derivatives of the optimal model coefficients do not
contribute to the derivative of the profiled objective.

The objective caches `pi`, fitted models, residuals, loss, and gradient. This
avoids recomputing a gradient when only a line-search value is requested and
allows line-search candidates to reuse the current fitted models.

### 7.3 Unit-normal constraint

Without fixing the norm of `w`, rescaling `(w,z)` changes the effective sigmoid
temperature and makes the parameterization non-identifiable. ART constrains
`w` to the unit sphere. The Euclidean gradient is projected onto the tangent
space:

```text
g_w_tan = (I - w w^T) grad_w J.
```

The `z` component is unchanged. The descent direction is minus this projected
gradient, and a candidate is retracted to the sphere by renormalizing its `w`
component after the step.

### 7.4 Armijo line search

For search direction `p`, the line search seeks an `alpha` satisfying

```text
J(R_theta(alpha p))
    <= J(theta) + c * alpha * <grad J(theta), p>,
```

where `R` is the sphere retraction. For the projected steepest-descent direction,
the directional derivative is `-||grad_projected J||^2` up to floating-point
error.

The initial line-search step can be fixed or controlled by `AdaptiveAlpha`:

- zero backtracks: grow the next initial step;
- a moderate number: start near the scale implied by the accepted step;
- many backtracks: recover conservatively from the accepted step;
- failed search: shrink the proposed initial step;
- always clip the proposal to `[alpha_min, alpha_max]`.

With adaptive control, a failed line search may be retried several times at a
smaller initial scale. If the controller is already at `alpha_min` and still
fails, the optimizer stops with `alpha_min_reached`; an accepted step at that
scale does not itself stop the optimizer. Saturation at either controller bound
is stored as a warning.

The gradient stopping condition is

```text
||grad_projected J(theta_k)||
    <= grad_atol + grad_rtol * ||grad_projected J(theta_0)||.
```

### 7.5 Frozen-model line search

The current default is `refit_during_line_search=False`. During backtracking,
the candidate `pi` is recomputed, but the current left/right models and residuals
are frozen. Once a candidate is accepted, both models are refit and the true
stored loss and gradient are recomputed.

This saves many weighted fits, but Armijo then guarantees decrease only for a
frozen-model surrogate. The refitted objective can increase after acceptance.
Consequences include misleading zero-backtrack growth, large proposed step
sizes, and loss histories that are not monotone. This behavior has been observed
in difficult Rastrigin nodes and currently prevents a standard convergence proof
for the default optimizer.

Setting `refit_during_line_search=True` is closer to the intended profiled
objective but can be much more expensive because every backtrack refits two
models.

### 7.6 Initialization, restarts, and hardening

Each restart samples a random unit normal and sets `z` to the median projection
of node data, so initialization starts with an approximately even empirical
split. All configured restarts run and the valid result with lowest hard
training MSE is selected.

After optimization, ART hardens the gate at `w^T x-z = 0`, checks the required
minimum count on both sides, and fits fresh unweighted child models. The hard
split result reports MSE, gain, side counts, convergence state, and histories.

Minimum side size is currently checked only on the final hard soft-oblique
partition, not after every gradient step. A soft optimization can therefore
drift into a saturated gate with all points effectively on one side and only be
rejected at the end.

## 8. Hinge Affine Splitter (HRT)

The hinge splitter is restricted to affine models. It fits either

```text
h(x) = max(theta_1^T x_aug, theta_2^T x_aug)
```

or the corresponding minimum. In `both` mode it tries both choices and keeps the
lowest-loss valid result.

Initialization first samples a random unit normal `v` and sets its offset to the
median projection of the node data. Thus the initial boundary

```text
delta_0 proportional to [v, -median(X v)]
```

starts with an approximately balanced hard partition. The two hinge pieces are
initialized symmetrically around the parent affine fit. `init_scale` changes the
magnitude of their difference, but not the geometry of this initial median
hyperplane.

Each iteration:

1. assigns points to the currently active affine piece;
2. rejects an intermediate partition below the side-size requirement;
3. fits an affine ridge model to each active set;
4. applies the damped update

```text
theta_j <- theta_j + mu * (theta_j_OLS - theta_j);
```

5. stops only when the active partition is unchanged and the relative parameter
   step is small.

Writing `delta = theta_1-theta_2`, the two updates imply exactly

```text
delta_new = (1-mu) delta_old + mu (theta_1_OLS-theta_2_OLS).
```

The induced boundary is `delta^T x_aug = 0`; its normal and offset are normalized
together before being returned. The final hard left/right models are fresh fits
through the shared ridge-solver interface, independent of max/min hinge-piece
labels. HRT stores conditioning and auto-solver histories in the same style as
the soft splitter.

HRT is best viewed as a fixed-point heuristic. The current code does not prove
that every update decreases hinge loss, excludes cycles, or converges globally.
It is also not generic over polynomial or kernel leaf models.

## 9. Sampling

### 9.1 Standard hit-and-run

For a polytope

```text
R = {x : A x <= b},
```

hit-and-run samples a direction `u`, computes the feasible chord

```text
I(x,u) = {t : A(x+t u) <= b},
```

draws `t` uniformly from that interval, and moves to `x+t u`. The implementation
supports burn-in and fixed thinning. With ordinary directions,
`u = z/||z||` for `z ~ N(0,I)`.

The region operations are dense. One transition costs approximately `O(m d)`
for `m` polytope constraints. A depth-`k` leaf has the root box constraints plus
roughly `k` split constraints.

### 9.2 Covariance-shaped directions

The builder's `isotropic_sampling=True` option is more accurately described as
covariance-shaped hit-and-run:

1. run an unthinned pilot chain of length `50*d` by default;
2. estimate the pilot sample covariance `Sigma = Q Lambda Q^T`;
3. floor eigenvalues at

```text
lambda_floor = eta * trace(Lambda) / d;
```

4. draw directions proportional to

```text
u_raw = Q Lambda_safe^(1/2) z,    z ~ N(0,I),
u     = u_raw / ||u_raw||.
```

This favors elongated directions inferred from the region and is intended to
improve mixing in skewed cells. The sampler floors eigenvalues internally; the
builder separately applies the same floor only to calculate condition numbers
and warnings. Pilot oracle values are never requested. Pilot points are not
currently reused as node training samples; final sampling starts at the last
pilot state.

If any raw pilot covariance eigenvalue is at or below the trace-scaled floor,
the node receives the sampling warning
`covariance_eigenvalue_floor_saturated`. This is not a sampling failure. It says
that the direction preconditioner had to raise at least one small variance,
which may reflect a genuinely thin cell, a poorly mixed pilot, or a noisy
finite-sample covariance estimate. The post-floor covariance condition estimate
is stored separately.

For a fixed positive-definite direction distribution that is symmetric under
`u -> -u`, exact uniform sampling along each chord should preserve the uniform
target. The data-dependent pilot makes the transition kernel adaptive before it
is frozen, so the conditional argument and finite-chain bias still need to be
written carefully.

The covariance calculation costs `O(N_pilot d^2 + d^3)`. This may dominate in
high dimension even though it uses no oracle queries.

### 9.3 ACF-based thinning estimator

**Experimental and not yet integrated into `RegressionTreeBuilder`.**

`sampling.py` can estimate a thinning lag from a consecutive pilot chain using
coordinate probes, random linear probes, or both. For scalar probe series `Z_t`,
the implemented estimate is

```text
gamma_hat(k) = (1/(N-k)) sum_{t=1}^{N-k}
                 (Z_t-Z_bar)(Z_{t+k}-Z_bar),

gamma_hat(0) = (1/N) sum_{t=1}^N (Z_t-Z_bar)^2,

rho_hat(k) = gamma_hat(k) / gamma_hat(0).
```

Each probe is centered and scaled once by its maximum absolute centered value.
The scaling cancels in the correlation ratio and prevents overflow/underflow.
Constant probes return zero. The ACF is signed and is not clipped to `[-1,1]`;
finite-sample normalization can produce small excursions outside that interval.

Optional whitening centers and globally scales the chain, estimates its sample
covariance, floors its spectrum relative to the maximum eigenvalue, and applies
`Sigma^(-1/2)` before probes are formed.

Given candidate lags, the estimator computes the maximum absolute ACF over all
probes and chooses the first candidate whose configured stability window stays
below a threshold. If no window passes, it returns the last candidate. For
powers-of-two candidates, the stability window is over neighboring candidates,
not consecutive integer lags.

The current direct computation costs approximately `O(N q L)` for chain length
`N`, number of probes `q`, and number of candidate lags `L`, in addition to
forming probe series. A proposed default such as `N=1000*d`, `q=2*d`, and
`L=200` could be a serious non-oracle bottleneck at high dimension. Incremental,
FFT-based, or batched lag evaluation should be considered before enabling it by
default.

Coordinate/random probes are geometric proxies. Low correlation for them does
not prove low correlation for the actual soft objective, residuals, or oracle
values.

## 10. Temperature Selection

Temperature has units of signed distance because `||w||=1`. It should therefore
adapt to the geometry and sample density of each node rather than be a universal
dimensionless constant.

The shared benchmark preset currently uses `strategy="fixed"`, median-nearest-
neighbor scaling, and `c=1`. Temperature tuning remains available but is not the
benchmark default.

Two scales are implemented:

```text
T = c * median nearest-neighbor distance,
```

and

```text
T = c * median pairwise distance * n^(-1/d).
```

The first is the default. Distances may use `cKDTree` or brute force. At most 512
points are used by default to cap pairwise work. The builder defaults to brute
force at dimension 20 and above because KD-tree performance degrades in high
dimension.

`TemperatureConfig` supports:

- `splitter`: use the splitter's absolute temperature unchanged;
- `fixed`: use a fixed scale constant `c`, recomputing local `T` from node data;
- `tune_root`: select `c` at the root and reuse that constant while recomputing
  each node's local geometric scale;
- `tune_node`: select `c` independently at every split node.

The current candidate constants are

```text
(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1).
```

Tuning uses a random fit/validation split of the existing node sample (20%
validation by default). Every candidate uses the same splitter initialization
seed for a fairer comparison. Invalid or builder-rejected candidates receive
infinite score; valid candidates are ranked by validation MSE. The selected
temperature is then rerun on all node data. Failure retries are disabled at a
node actively performing tuning, but are available in later `tune_root` child
nodes where the root-selected `c` is merely reused.

Current concerns:

- tuning reduces the fitting sample available to each candidate;
- it chooses by MSE even when the stopping metric is different;
- repeated candidate selection introduces validation-selection bias;
- no uncertainty or standard-error penalty is used;
- Euclidean distances are sensitive to anisotropic coordinate scaling;
- very small `T` can saturate the sigmoid and create nearly zero gradients;
- very large `T` can hide a genuine discontinuity by blending the models;
- tune-at-every-node can dominate training time.

A continuation strategy, starting smooth and annealing toward a geometry-based
lower temperature, is a promising alternative to a flat grid search.

## 11. Failure Handling and Diagnostics

### 11.1 Splitter failures

`SplitNotFoundError` carries a machine-readable reason, optional diagnostics,
retry count, prior failure reasons, and context. Important reasons include:

- `min_side_points`;
- `invalid_split`;
- `temperature_tuning_failed`.

Optimizer stop reasons are reported through `SplitResult`, including gradient
tolerance, maximum iterations, line-search failures, alpha saturation, and
non-descent directions. A maximum-iteration result may still be accepted if its
hard split is valid and improves MSE.

Warnings are orthogonal to node status. Accepted/rejected `SplitResult` warnings
include `small_init_grad` for a zero-iteration gradient-tolerance stop,
alpha-bound saturation, `max_iters`, and high model conditioning. Sampling keeps
its own warning tuple, including covariance-eigenvalue-floor saturation. A split
may be accepted with warnings, while a rejected leaf may retain both its
rejection status and diagnostic warnings. For a leaf created directly from
`SplitNotFoundError`, only small-initial-gradient and high-conditioning flags are
currently promoted to the top-level warning tuple; fuller alpha and optimizer
state may still be present in `splitter_metadata`.

### 11.2 Builder rejections

The builder separately rejects:

- nonfinite split summaries;
- insufficient absolute or relative MSE gain;
- selected numerical failures that made zero accepted iterations.

`max_retries_on_failure` can rerun a single-restart splitter with new `(w,z)`
initialization on `min_side_points` or insufficient gain. It reuses the same
`X,y`; no oracle calls are added. To control cost, retries are permitted only
when `splitter.n_restarts == 1` and are disabled while a node is actively tuning
temperature.

### 11.3 Stored data

With `store_diagnostics=True`, accepted split nodes store the chosen run's loss,
projected-gradient, accepted-step, backtrack, and related optimizer metadata.
Leaves created by a split failure or rejection retain diagnostics from the last
failed attempt. If internal splitter restarts find a valid split, the best valid
result is stored; if all fail, the last failure is retained.

With `store_samples=True`, every node additionally stores its full `X,y`. This
can dwarf all other metadata and should remain off for large trees.

The tree supports depth-first node/leaf iteration, lookup by `node_id`, retrieval
of a complete root-to-node path, and batched routing. Joblib artifacts preserve
the tree, model weights, node metadata, diagnostics, and a run configuration.
Joblib should only load trusted files and is not a stable, language-neutral
interchange format.

The 2D and high-dimensional benchmark harnesses save `tree.joblib`, a compact
`node_diagnostics.csv`, and `report.txt`; plots depend on the harness. Reports
include prediction metrics, tree/failure counts, timings, and the estimated
oracle minimum, maximum, and span over independent uniform test points. The
shared high-dimensional report additionally includes pointwise-relative
quantiles. These range estimates are not exact extrema over the continuous
domain, and test/plot oracle evaluations do not increment the tree's training
query count.

`tree_diagnosis.ipynb` loads trusted artifacts, filters nodes by status, warning,
and depth, ranks or randomly selects them, and plots optimizer, solver-
conditioning, and root-to-node error diagnostics.

### 11.4 Coarse build timing

`profile_build_timing=True` enables aggregate timing and is off by default. The
profile records total build time, sampling time, splitter time, and soft-
objective model-refit time; each is reported as a percentage of total build
time. Model-refit time is a nested subset of splitter time, while `other` is
computed only from the top-level sampling and splitter categories. HRT reports
sampling and splitter timing without a separate internal refit category.

The model-refit timer covers only `_fit_models()`. It does not include sigmoid
evaluation, predictions and residual construction after a fit, loss/gradient
evaluation, frozen-model Armijo candidates, diagnostic bookkeeping, or final
hard child fits. In current stress tests these supposedly cheap per-candidate
operations can collectively exceed solve time because millions of optimizer and
backtracking evaluations are performed. The profile is for broad attribution,
not a complete call-level profiler.

## 12. What Works Today

The prototype can currently:

- build oblique regression trees on boxes and bounded convex polytopes;
- query scalar deterministic oracles in scalar or vectorized form;
- fit affine or explicit polynomial ridge leaves;
- optimize sigmoid-gated oblique splits with projected gradients and Armijo
  backtracking;
- use the affine HRT heuristic as an alternative splitter;
- inherit and top up child samples without re-querying inherited points;
- use exact uniform root samples for boxes and hit-and-run for child polytopes;
- shape hit-and-run directions using a pilot covariance;
- scale or tune soft temperature from node geometry;
- enforce depth, side-size, and empirical MSE-gain safeguards;
- retry selected failed split attempts without additional oracle evaluations;
- count training oracle queries;
- route inference in batches rather than point by point;
- preserve detailed diagnostics and save/load trusted trees;
- optionally profile aggregate sampling, splitter, and soft model-refit time;
- run reproducible 2D and higher-dimensional affine/polynomial tree benchmarks
  with shared presets and reporting.

Benchmark functions currently cover quadratics, Gaussians, Gaussian mixtures,
Rosenbrock, Rastrigin, plane waves, and sphere-separated piecewise polynomials.
Gaussian mixtures are currently part of the 2D harness but intentionally omitted
from the high-dimensional harness.

## 13. Known Flaws and Risks

### 13.1 Objective-gradient inconsistency

This is the most important mathematical issue in the soft splitter.

The reported outer loss contains weighted squared residuals but not a ridge
penalty. The inner models are nevertheless ridge fits. Moreover, the effective
ridge coefficient depends on the sigmoid weights through `trace(Phi^T W Phi)`,
and the weight floor changes the weights used in the inner solve. Therefore the
inner coefficients are not exact minimizers of the reported outer loss, and the
implemented residual-only envelope gradient is not exactly its derivative.
The shared benchmark preset currently sets `ridge=0`, which removes the ridge-
penalty part of this mismatch for those runs, but the positive weight floor can
still differ from the `pi` used in the reported loss when gates saturate.

Possible resolutions:

1. Define a coherent penalized profiled objective, use a node-fixed ridge scale,
   and include the penalty in values used by optimization.
2. Retain weight-dependent ridge scaling but derive and implement all additional
   derivative terms.
3. Treat ridge purely as tiny numerical jitter, quantify gradient error, and
   state an approximation theorem.
4. Use a stable unregularized QR/SVD solve when rank permits and regularize only
   explicitly singular cases.

The sigmoid also clips logits to `[-50,50]`, while the gradient uses
`pi(1-pi)` as if no hard clipping occurred. The discrepancy is numerically tiny
in the saturated range but should be removed for a clean proof, for example by
using a stable unclipped logistic implementation.

### 13.2 Frozen line-search mismatch

The default line search tests a frozen-model surrogate but adapts step size as if
it tested the true profiled objective. Accepted refitted loss can increase, so
classical Armijo convergence results do not apply. A robust compromise could
perform cheap frozen screening followed by one true refit-and-acceptance check;
if that check fails, the algorithm must shrink or invoke a trust-region rule.

### 13.3 Degenerate soft gates

The optimization has no balance constraint or penalty. `z` is unbounded, and a
sigmoid can saturate with nearly all mass on one side. Then `pi(1-pi)` and the
gradient become tiny, creating false convergence even though the final hard
split violates minimum side size. Random restarts help but do not solve the
geometry.

Candidate remedies include quantile constraints on `z`, a differentiable
effective-mass barrier, a constrained parameterization of the offset, or
retaining the best optimizer iterate that satisfies the final hard side-count
constraint.

This is not only a plane-outside-the-cell failure. In the spherical piecewise
benchmark, failed planes were observed to cross their cells and closely
approximate local tangents to the true curved interface, yet isolate only one or
two of 150 training points. The corresponding geometric caps could have
nonzero measure and the regression solves could be well conditioned. The
unconstrained soft loss simply preferred the small residual-rich side, and
restarts were attracted to the same invalid solution.

### 13.4 Optimistic stopping and split selection

Node error is measured on training data. Split gain is also measured on the data
used to optimize the split and fit both children. Both are optimistic, especially
for flexible polynomials, many restarts, and temperature grids. A node can stop
prematurely or accept a noise-fitting split.

A research-quality implementation needs holdout, cross-fitting, bootstrap, or
confidence-bound options, with careful accounting for their extra oracle cost.

### 13.5 MCMC dependence

Inherited and hit-and-run samples are neither independent nor identically
generated in a simple finite-sample sense. Random train/validation splits ignore
chain order. Fixed thinning is global, while region geometry changes by node.
The existing ACF estimator is only a proxy and is not integrated into the tree.

### 13.6 Numerical linear algebra

- The low-level model default `normal` mode uses normal equations and therefore
  squares the design condition number; shared benchmark presets use `auto`
  instead.
- Raw polynomial features can have extreme dynamic range.
- Ridge scaling can become very small in a low-energy design.
- The code has no standardized input/feature scaling policy.
- Auto mode provides Cholesky-to-QR fallback and accepted-iterate conditioning
  histories, but its `dpocon`-derived estimate is only a 1-norm proxy for the
  augmented design condition and the `1e-10` threshold is empirical.
- Pointwise relative metrics can explode near `y=0` and depend strongly on their
  denominator floor. The stopping criterion and raw diagnostic error are not yet
  separated.

### 13.7 Geometry and feasibility

- Regions accumulate one dense constraint per depth and are never simplified.
- Feasible-start fallback uses rejection from an inferred/provided box and can
  fail badly for tiny-volume polytopes.
- There is no linear-programming feasibility or interior-point initializer.
- No minimum geometric child volume is checked; sample count is only a proxy.
- Floating-point boundary tolerances are not centrally calibrated.

### 13.8 Scalability

- Polynomial dimension grows combinatorially.
- Soft optimization refits two models per accepted iteration and possibly per
  backtrack.
- Temperature tuning multiplies splitter work by grid size.
- Restarts, retries, and children are serial.
- Covariance pilots and eigendecompositions can dominate at high dimension.
- Direct ACF evaluation can be more expensive than sampling itself.
- Frozen-model backtracking avoids solves but still repeatedly allocates
  candidate state and evaluates projections, logits, sigmoids, reductions, and
  losses over all node samples.
- There is no global oracle budget, leaf budget, pruning pass, or parallel build.

### 13.9 API and packaging gaps

- The repository does not yet have a polished installable package configuration,
  public API guide, version policy, changelog, or migration-aware artifacts.
- Failure reasons are strings rather than a stable enum or typed status object.
- Kernel ridge is not soft-split compatible.
- There is no multi-output, classification, noisy-oracle, uncertainty, online,
  or partial-fit support.
- There is no sklearn-style estimator interface or systematic callback/logging
  interface.
- Joblib persistence is Python-version/dependency sensitive and unsafe for
  untrusted artifacts.

## 14. Theoretical Work Needed

A paper should separate what can be proved for an idealized algorithm from what
is used as an engineering heuristic.

### 14.1 Define the statistical problem precisely

Specify:

- deterministic versus noisy oracle;
- target measure on the domain;
- allowed model classes and regularization;
- whether the objective is global integrated MSE, relative error, or a uniform
  norm;
- whether oracle-query count, computation, or both are optimized;
- assumptions on domain geometry and function regularity.

Without this, the meaning of "adaptive" and "accurate" remains ambiguous.

### 14.2 Correctness of the soft objective and gradient

For a mathematically coherent objective, prove:

1. existence and uniqueness of each weighted inner fit;
2. differentiability of the profiled objective;
3. the envelope/variable-projection gradient formula;
4. how regularization and weight floors enter both objective and derivative;
5. scale identifiability under `||w||=1`;
6. equivalence between the projected gradient and the Riemannian gradient on
   `S^(d-1) x R`.

The current implementation must be adjusted before this theorem is literally
true.

### 14.3 Convergence of the split optimizer

Under a smooth exact profiled objective, compact/bounded offset set, Lipschitz
gradient, and a valid retraction, establish that projected/Riemannian gradient
with Armijo produces sufficient descent and that accumulation points are
stationary. This is a local result; the split objective is nonconvex, so a global
minimum guarantee is not realistic without much stronger structure.

The frozen-model line search needs a separate inexact-gradient or majorization
analysis, or it should be changed so true-objective acceptance is enforced.

### 14.4 Split balance and nondegeneracy

Formulate side balance as a constraint on empirical or population mass. Prove
that initialization and updates remain feasible, or that a barrier prevents
saturation. Relate empirical side counts to true child volume with a
concentration bound.

### 14.5 Hit-and-run validity

Prove or cite, under explicit assumptions:

- invariance/reversibility of uniform measure for ordinary and
  covariance-shaped symmetric direction distributions;
- irreducibility after eigenvalue flooring;
- the conditional-uniform argument for inherited child seeds;
- finite burn-in/mixing error from a nonstationary seed;
- the effect of pilot-estimated covariance on the frozen sampling kernel.

Known hit-and-run mixing theorems can likely be cited, but constants and
rounding assumptions must match this implementation.

### 14.6 Dependent-sample error bounds

Derive or cite concentration bounds for node MSE, stopping metrics, and split
gain under a geometrically mixing Markov chain. Express uncertainty through
effective sample size or integrated autocorrelation time. The challenge is
greater because the tree, split, temperature, and future regions are all chosen
adaptively from the same data.

A practical first paper may prove statistical statements under iid uniform node
samples and treat MCMC effects empirically, clearly labeling that limitation.

### 14.7 Temperature asymptotics

Temperature is both an optimization smoothing parameter and a statistical
bandwidth. A useful theory should identify conditions such as:

- `T_n -> 0` so the soft objective approaches the hard partition objective;
- enough samples remain in the transition slab to estimate a stable gradient;
- the optimizer does not enter sigmoid saturation prematurely;
- a geometry-based estimate is invariant to domain rescaling.

Nearest-neighbor spacing scales roughly with local sample density, but that fact
alone does not prove optimality for boundary estimation. The appropriate rate
may depend on dimension, boundary codimension, model error contrast, and oracle
noise.

### 14.8 Recursive/global error control

For squared error, global risk decomposes as a leaf-mass-weighted sum of local
risks. Use that structure to show when local stopping guarantees a global target.
Relative L2 and maximum pointwise error require different arguments. In
particular, an unweighted per-leaf tolerance does not automatically control the
global metric unless leaf masses and denominators are accounted for.

### 14.9 Query and computational complexity

Give upper bounds in terms of dimension, effective model dimension, depth,
number of leaves, minimum side fraction, thinning, and optimizer iterations.
Separate:

- new oracle evaluations;
- inherited values;
- geometry-only Markov transitions;
- weighted solves;
- feature construction;
- inference cost.

The current fixed target allows a simple query accounting identity, but retry,
tuning, and pilot costs must be reported separately from oracle cost.

### 14.10 Approximation and consistency

The strongest long-term result would characterize functions for which oblique
piecewise-polynomial trees achieve favorable approximation rates, and show
consistency as query budget and allowable tree size increase. Candidate classes
include piecewise smooth functions with low-complexity interfaces and smooth
functions with anisotropic curvature.

### 14.11 HRT analysis

For the hinge fixed-point method, determine whether stable partitions imply a
fixed point, when damped updates decrease an objective, whether cycles can occur,
and what can be guaranteed over finitely many active sets. Otherwise present HRT
only as a cited baseline/heuristic.

## 15. What Would Make This a Strong Paper

### 15.1 A focused contribution

Avoid claiming every implemented option as a separate contribution. A coherent
paper could center on:

> Oracle-efficient adaptive surrogate construction using oblique partitions,
> variable-projection split optimization, and geometry-aware sampling.

The paper should make clear which components are novel, which are adaptations of
known oblique/hinge trees and hit-and-run, and which are engineering choices.

### 15.2 A tractable theory package

A credible first theory contribution could contain:

1. a corrected differentiable profiled objective and exact gradient theorem;
2. local stationary-point convergence for exact-refit Riemannian Armijo;
3. uniform invariance of covariance-shaped hit-and-run after the pilot is frozen;
4. global MSE decomposition and an oracle-query bound for a fixed tree;
5. iid finite-sample split/leaf error bounds under a finite-dimensional model.

Trying to prove full adaptive-tree consistency with MCMC samples immediately may
obscure the core contribution.

### 15.3 Baselines

Compare against methods appropriate to both tree construction and surrogate
modeling:

- axis-aligned CART/model trees;
- established oblique regression trees;
- hinge regression trees;
- one global polynomial/kernel model;
- MARS or related piecewise regression;
- Gaussian processes, neural surrogates, or adaptive sparse grids where scale
  permits.

Every comparison should use oracle queries as the primary x-axis, with wall time,
memory, number of leaves, and depth also reported.

### 15.4 Experiments and ablations

Use many seeds and confidence intervals. Include:

- exact piecewise affine/polynomial functions with known hyperplanes;
- curved interfaces such as sphere-separated pieces;
- smooth anisotropic functions;
- oscillatory functions such as plane waves and Rastrigin;
- localized peaks and Gaussian mixtures;
- noisy-oracle variants if noise is in scope;
- moderate/high-dimensional tests with controlled effective dimension;
- at least one real scientific simulator or public surrogate benchmark.

Measure prediction MSE/relative errors, boundary angle/offset error when known,
partition misclassification, leaves, query count, optimizer failures, sampling
ESS, and runtime.

Essential ablations:

- soft splitter versus HRT and axis-aligned split;
- exact-refit versus frozen-model line search;
- fixed versus tuned/annealed temperature;
- ordinary versus covariance-shaped hit-and-run;
- fixed versus ACF-selected thinning;
- polynomial degree and ridge solver;
- sample multiplier;
- side-balance constraint;
- restarts and failure retries;
- split-gain thresholds and post-pruning.

Include honest failure-case visualizations. The saturated-gate/flat-objective
behavior is scientifically useful evidence, not merely a bug to hide.

### 15.5 Reproducibility

Provide pinned environments, deterministic seed handling, serialized run
configuration, scripts that regenerate every table/figure, machine-readable raw
results, and documented hardware/timing methodology.

## 16. What Would Make This a Useful Library

### Priority 0: correctness

- Make the optimized objective, ridge solve, and gradient mathematically
  consistent.
- Add a true-objective safeguard to frozen-model line search.
- Add a balance-aware parameterization or constraint for `(w,z)`.
- Benchmark normal/QR/SVD/auto modes and calibrate the auto `rcond` threshold,
  warning threshold, and solver-switching behavior across model classes.
- Add independent validation or confidence-aware stopping and split acceptance.
- Separate the acceptance/stopping metric from raw fit-error diagnostics,
  including an absolute-or-relative pointwise rule near oracle zeros.
- Integrate ACF thinning only after profiling and validating it.
- Add hard query, node, and wall-time budgets.

### Priority 1: API and reliability

- Add `pyproject.toml`, package metadata, semantic versioning, README, API docs,
  tutorials, license, and changelog.
- Define stable public protocols and typed failure/status enums.
- Provide a concise estimator API with validated configuration objects.
- Add structured logging/callbacks instead of ad hoc output.
- Add higher-level node filtering helpers for diagnostics, including queries
  such as "all leaves above error threshold" and "failed leaves by reason".
- Version persistence schemas and store JSON-safe run configuration alongside
  joblib weights; document trusted loading.
- Add weighted kernel support or explicitly remove it from supported ART models.
- Centralize numerical tolerances and random-state behavior.

### Priority 2: scale and scope

- Parallelize independent temperature candidates, restarts, and child builds.
- Cache/reuse geometry and feature work across compatible operations.
- Simplify redundant polytope constraints and add LP-based feasible starts.
- Add pruning and global leaf/query allocation.
- Support multi-output regression and noisy-oracle uncertainty.
- Add sample deduplication/caching where exact duplicate oracle queries can occur.
- Profile batched inference and feature transforms on large trees.
- Add optional compact diagnostics so histories do not dominate artifacts.

### Testing needed

- Unit tests for every failure reason and retry path.
- Finite-difference tests of the corrected soft gradient over multiple model
  classes and temperatures.
- Property tests for region splitting, routing, sample feasibility, and prepared
  design row alignment.
- Statistical tests for sampler moments and invariance with fixed tolerances.
- Regression tests for serialization and old artifact versions.
- Numerical tests for rank-deficient and badly scaled designs.
- Performance benchmarks with explicit acceptable regressions.

## 17. Open Research Questions

1. Should ART target deterministic approximation only, or noisy regression too?
2. Is uniform domain measure the primary objective, or should arbitrary target
   densities be first-class?
3. What is the simplest coherent regularized variable-projection objective?
4. Can a frozen-model line search be justified as majorization/minimization, or
   should it be replaced by a true-loss acceptance safeguard?
5. What balance constraint prevents saturation without excluding useful small
   regions?
6. Should temperature be selected by validation, continuation, gradient quality,
   or a joint criterion?
7. Which observables best determine H&R thinning for the split objective?
8. Is thinning worthwhile versus running a longer unthinned chain and accounting
   for effective sample size?
9. When is a zero-burn-in inherited child seed sufficiently close to stationary?
10. Can samples be allocated adaptively using fit uncertainty or split-gradient
    instability rather than a fixed `50*p_eff` target?
11. Should split acceptance use the user's error metric, MSE, a penalized risk, or
    a confidence bound?
12. How should leaf probability mass enter stopping and global error control?
13. Can oblique piecewise-polynomial trees achieve provably better rates than
    axis-aligned trees for low-complexity interfaces?
14. Which diagnostics best predict that a failed low-depth node deserves more
    samples, a new temperature, or a new initialization?
15. At what dimensions do covariance pilots, polynomial features, and oblique
    optimization stop being competitive?
16. How should absolute and relative tolerances be combined near oracle zeros
    without hiding the raw relative-error distribution in diagnostics?

## 18. Recommended Near-Term Sequence

1. Correct and finite-difference-test the regularized profiled objective.
2. Add true-loss validation after frozen-model candidate acceptance.
3. Add a balance-aware offset constraint or best-valid-iterate fallback and use
   the known spherical-interface failures as regression cases.
4. Calibrate the implemented `auto` solver's reciprocal-condition threshold and
   high-conditioning warning against explicit QR/SVD runs.
5. Separate stopping from raw fit diagnostics, then add a held-out node
   stopping/split-gain option and compare query cost.
6. Integrate adaptive thinning behind a configuration object, recording selected
   lag, threshold failure, ESS proxy, and pilot cost.
7. Use the coarse build profile to target selective finer measurements of
   feature work, frozen Armijo candidates, accepted-state evaluation, and hard
   fits without leaving pervasive timers in hot loops.
8. Freeze a reproducible benchmark suite and run the core ablations.
9. Write the theory for the corrected exact-refit algorithm.
10. Package the stable subset as a documented public API.

## 19. Guidance for Future Agents

- Read the implementation before changing an algorithm; the old notebooks are
  conceptual references, not specifications.
- Do not claim a feature is supported merely because a class exists. Verify the
  builder, splitter, weighted-fit, sample-budget, and persistence contracts.
- Keep raw routing coordinates distinct from transformed model features.
- Prefer the shared presets for comparable benchmark runs, but distinguish their
  defaults from the lower-level class defaults.
- Reuse prepared designs across parent fitting, split iterations, retries,
  temperature candidates, and hard child fits.
- Preserve oracle-query accounting. Pilot geometry samples are free only in the
  oracle-count sense, not in runtime.
- Preserve diagnostics for failed leaves when diagnostics are enabled.
- Distinguish soft objective loss, frozen line-search loss, hard split MSE, node
  stopping error, raw diagnostic fit error, and external test error; they are
  different quantities.
- Add tests with each numerical change, especially around model refits and
  cached residuals.
- Avoid silently weakening safeguards to make a benchmark finish. A failed node
  is data about the method.
- Keep research heuristics clearly labeled until theory or strong empirical
  validation supports them.

ART already has the core of a serious experimental system: adaptive querying,
oblique partitions, generic weighted local models, geometry-aware sampling, and
rich diagnostics. Its most important next step is not adding more benchmark
functions or optimizer switches. It is making the optimized mathematical object,
the gradient, and the line-search acceptance rule agree. Once that foundation is
solid, the sampling and recursive error-control story can become both a strong
paper contribution and a dependable library.
