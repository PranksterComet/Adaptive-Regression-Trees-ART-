"""Hybrid Gram backend: CPU integration and optional GPU numerical checks."""

import builtins

import numpy as np
import pytest

from art._cuda_hybrid_gram import CudaWeightedGram, cuda_status
from art.models import (
    AffineRidgeModel,
    PolynomialRidgeModel,
    WeightedNormalEquations,
    solve_weighted_ridge,
)
from art.splitters import SoftObliqueSplitter


@pytest.mark.parametrize("ridge", [0.0, 1e-5])
@pytest.mark.parametrize("singular", [False, True])
def test_precomputed_system_preserves_cpu_solve_and_fallback(ridge, singular):
    rng = np.random.default_rng(117)
    phi = rng.normal(size=(80, 5))
    if singular:
        phi[:, -1] = phi[:, 0]
    y = rng.normal(size=80)
    weights = rng.uniform(0.0, 1.0, 80)
    weights[:3] = 0.0
    q = np.maximum(weights, 1e-12)
    system = WeightedNormalEquations(phi.T @ (phi * q[:, None]), phi.T @ (y * q))
    expected = solve_weighted_ridge(phi, y, weights, ridge, solver="auto")
    actual = solve_weighted_ridge(
        phi, y, weights, ridge, solver="auto", normal_equations=system
    )
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
    assert actual.solver_used == expected.solver_used
    assert actual.fallback_reason == expected.fallback_reason
    assert actual.ridge_effective == expected.ridge_effective
    if singular and ridge == 0.0:
        assert actual.solver_used == "qr"


@pytest.mark.parametrize("problem", ["gram", "rhs", "cholesky"])
def test_unreliable_complement_is_rebuilt_before_solving(problem):
    rng = np.random.default_rng(118)
    phi = rng.normal(size=(60, 4))
    y = rng.normal(size=60)
    q = np.ones(60)
    good = WeightedNormalEquations(phi.T @ phi, phi.T @ y)
    calls = []

    def rebuild():
        calls.append(True)
        return good

    gram = np.zeros((4, 4)) if problem == "cholesky" else np.eye(4)
    derived = WeightedNormalEquations(
        gram, np.zeros(4),
        subtraction_bound=0.0 if problem == "rhs" else 1.0,
        rebuild=rebuild,
        subtraction_rhs_bound=1.0 if problem == "rhs" else 0.0,
    )
    actual = solve_weighted_ridge(
        phi, y, q, 0.0, solver="auto", normal_equations=derived
    )
    expected = solve_weighted_ridge(phi, y, q, 0.0, solver="auto")
    assert calls == [True]
    assert actual.solver_used == "cholesky"
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)


def test_default_cpu_splitter_does_not_import_cupy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("CPU splitter must not import CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    rng = np.random.default_rng(119)
    X = rng.uniform(-1, 1, (100, 2))
    y = np.where(X[:, 0] > 0, X[:, 1], -X[:, 1])
    params = dict(
        model_template=AffineRidgeModel(), temperature=0.3,
        random_state=3, max_iters=3,
    )
    original = SoftObliqueSplitter(**params).split(X, y)
    explicit = SoftObliqueSplitter(**params, optimizer="cpu").split(X, y)
    np.testing.assert_array_equal(original.w, explicit.w)
    assert original.z == explicit.z
    assert original.loss == explicit.loss


def test_workspace_rejects_different_same_shape_data():
    phi = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    workspace = CudaWeightedGram(phi, y, 1e-12)
    workspace.validate(phi.copy(), y.copy(), 1e-12)
    with pytest.raises(ValueError, match="different design"):
        workspace.validate(phi + 1, y, 1e-12)
    with pytest.raises(ValueError, match="different design"):
        workspace.validate(phi, y + 1, 1e-12)
    assert workspace._cp is None  # Validation must not initialize a GPU.


def test_cuda_objective_prepares_polynomial_features_without_loading_cupy():
    from art.objectives import SoftObliqueRidgeObjective, make_cuda_workspace

    rng = np.random.default_rng(124)
    X = rng.normal(size=(25, 3))
    y = rng.normal(size=25)
    model = PolynomialRidgeModel(degree=2)
    objective = SoftObliqueRidgeObjective(1.0, model, X, y, optimizer="hybrid_gram")
    workspace = make_cuda_workspace(model, X, y, None, objective.weight_floor)
    assert objective.prepared_design.matrix.shape == (25, 10)
    np.testing.assert_array_equal(workspace.design, objective.prepared_design.matrix)
    assert workspace._cp is None


def test_builder_reuses_workspace_across_retries_and_does_not_serialize_it(
    monkeypatch, tmp_path
):
    import joblib
    import art.builder as builder_module
    from art.domain import BoxDomain
    from art.models import augment_features
    from art.splitters import SplitNotFoundError

    created = []
    attempts = []

    class CpuAssemblyWorkspace:
        """Exercise CUDA-path orchestration without emulating CUDA kernels."""

        def __init__(self, phi, y, floor):
            self.phi, self.y, self.floor = phi, y, floor

        def validate(self, phi, y, floor):
            np.testing.assert_array_equal(phi, self.phi)
            np.testing.assert_array_equal(y, self.y)
            assert floor == self.floor

        def form_pair(self, pi):
            return tuple(
                WeightedNormalEquations(
                    self.phi.T @ (self.phi * q[:, None]),
                    self.phi.T @ (self.y * q),
                ) for q in (np.maximum(1 - pi, self.floor), np.maximum(pi, self.floor))
            )

    def workspace_factory(model, X, y, prepared, floor):
        workspace = CpuAssemblyWorkspace(augment_features(X), y, floor)
        created.append(workspace)
        return workspace

    original_split = SoftObliqueSplitter.split

    def retry_once(self, *args, **kwargs):
        attempts.append(kwargs["gram_workspace"])
        if len(attempts) == 1:
            raise SplitNotFoundError("min_side_points", "force one retry")
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(builder_module, "make_cuda_workspace", workspace_factory)
    monkeypatch.setattr(SoftObliqueSplitter, "split", retry_once)
    model = AffineRidgeModel(ridge=0.0, solver="auto")
    builder = builder_module.RegressionTreeBuilder(
        domain=BoxDomain(np.array([[-1., 1.], [-1., 1.]])),
        oracle=lambda x: (1.0 if x[0] > 0 else -1.0) + 0.2 * x[1],
        model_template=model,
        splitter=SoftObliqueSplitter(
            model.clone(), temperature=0.3, max_iters=3,
            optimizer="hybrid_gram", min_side_fallback=True,
        ),
        sample_count=60, max_depth=1, error_threshold=0.0,
        max_retries_on_failure=1, random_state=5,
    )
    tree = builder.build().tree
    assert len(created) == 1
    assert attempts == [created[0], created[0]]
    path = tmp_path / "tree.joblib"
    joblib.dump(tree, path)  # A retained local workspace would not be pickleable.
    restored = joblib.load(path)
    X = np.array([[0.2, 0.1], [-0.7, 0.4]])
    np.testing.assert_array_equal(restored.predict(X), tree.predict(X))


@pytest.fixture(scope="module")
def gpu():
    available, reason = cuda_status()
    if not available:
        pytest.skip(f"CUDA hardware test: {reason}")


@pytest.mark.parametrize("gate", ["balanced", "few_clipped", "many_clipped", "zero", "one"])
@pytest.mark.parametrize("floor", [1e-12, 0.1])
def test_cuda_pair_matches_direct_floored_systems(gpu, gate, floor):
    rng = np.random.default_rng(120)
    phi = rng.normal(size=(80, 7))
    y = rng.normal(size=80)
    pi = rng.uniform(0.2, 0.8, 80)
    if gate == "few_clipped":
        pi[:4] = 0.0
        pi[4:8] = 1.0
    elif gate == "many_clipped":
        pi[:40] = 0.0
    elif gate == "zero":
        pi[:] = 0.0
    elif gate == "one":
        pi[:] = 1.0
    workspace = CudaWeightedGram(phi, y, floor)
    for gates in (pi, 1 - pi):  # Reuse allocations with genuinely new weights.
        systems = workspace.form_pair(gates)
        for system, raw in zip(systems, (1 - gates, gates)):
            q = np.maximum(raw, floor)
            np.testing.assert_allclose(
                system.gram, phi.T @ (q[:, None] * phi), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(system.rhs, phi.T @ (q * y), rtol=1e-12, atol=1e-12)
    assert workspace.setup_seconds > 0


def test_cuda_rebuilds_a_complement_with_cancelling_rhs(gpu):
    rng = np.random.default_rng(123)
    phi = rng.normal(size=(100, 5))
    y = np.zeros(100)
    y[:4] = 1e12
    pi = np.full(100, 0.1)
    pi[:4] = 1.0
    workspace = CudaWeightedGram(phi, y, 1e-12)
    system, _ = workspace.form_pair(pi)
    actual = solve_weighted_ridge(
        phi, y, 1 - pi, 0.0, solver="auto", normal_equations=system
    )
    expected = solve_weighted_ridge(phi, y, 1 - pi, 0.0, solver="auto")
    assert workspace.direct_rebuilds == 1
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("degree", [1, 2])
def test_cuda_splitter_matches_cpu(gpu, degree):
    rng = np.random.default_rng(121)
    X = rng.uniform(-1, 1, (150, 2))
    y = np.where(X[:, 0] > 0, 1 + X[:, 1] ** degree, -X[:, 1])
    model = (
        AffineRidgeModel(solver="auto") if degree == 1
        else PolynomialRidgeModel(degree=2, solver="auto")
    )
    params = dict(
        model_template=model, temperature=0.3, random_state=7,
        max_iters=4, min_side_fallback=True,
    )
    cpu = SoftObliqueSplitter(**params).split(X, y)
    cuda = SoftObliqueSplitter(**params, optimizer="hybrid_gram").split(X, y)
    np.testing.assert_allclose(cuda.w, cpu.w, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(cuda.z, cpu.z, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(cuda.predict(X), cpu.predict(X), rtol=1e-7, atol=1e-8)
    assert cuda.stop_reason == cpu.stop_reason


def test_cuda_rank_deficient_fit_uses_cpu_qr(gpu):
    rng = np.random.default_rng(122)
    phi = rng.normal(size=(90, 6))
    phi[:, -1] = phi[:, 0]
    y = rng.normal(size=90)
    pi = rng.uniform(0.1, 0.9, 90)
    systems = CudaWeightedGram(phi, y, 1e-12).form_pair(pi)
    for system, q in zip(systems, (1 - pi, pi)):
        result = solve_weighted_ridge(phi, y, q, 0.0, solver="auto", normal_equations=system)
        expected = solve_weighted_ridge(phi, y, q, 0.0, solver="qr")
        assert result.solver_used == "qr"
        np.testing.assert_allclose(
            phi @ result.coefficients, phi @ expected.coefficients,
            rtol=1e-8, atol=1e-9,
        )
