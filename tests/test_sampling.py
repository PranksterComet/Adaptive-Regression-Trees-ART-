import numpy as np
import pytest

import art.sampling as sampling_module
from art.domain import BoxDomain
from art.sampling import (
    BatchedHitAndRunSampler,
    HitAndRunSampler,
    NumbaBatchedHitAndRunSampler,
    ParallelNumbaBatchedHitAndRunSampler,
    autocorrelation_by_lag,
    estimate_thinning_from_chain,
    floor_covariance_eigenvalues,
    sample_covariance_eigendecomposition,
)


class _FixedHitAndRunStream:
    """Serve the same directions and chord uniforms through either RNG API."""

    def __init__(
        self,
        directions: np.ndarray,
        uniforms: np.ndarray,
        chain_major: bool = False,
    ) -> None:
        self.directions = np.asarray(directions, dtype=float)
        self.uniforms = np.asarray(uniforms, dtype=float)
        self.chain_major = chain_major
        self.direction_index = 0
        self.uniform_index = 0

    def normal(self, size: int | tuple[int, ...]) -> np.ndarray:
        shape = (size,) if isinstance(size, int) else tuple(size)
        dimension = self.directions.shape[1]
        if shape == (dimension,):
            result = self.directions[self.direction_index].copy()
            self.direction_index += 1
            return result

        if self.chain_major:
            n_chains, n_steps, requested_dimension = shape
        else:
            n_steps, n_chains, requested_dimension = shape
        assert n_chains == 1
        assert requested_dimension == dimension
        start = self.direction_index
        self.direction_index += n_steps
        selected = self.directions[start : self.direction_index]
        if self.chain_major:
            return selected[None, :, :].copy()
        return selected[:, None, :].copy()

    def random(self, size: tuple[int, ...]) -> np.ndarray:
        if self.chain_major:
            n_chains, n_steps = size
        else:
            n_steps, n_chains = size
        assert n_chains == 1
        start = self.uniform_index
        self.uniform_index += n_steps
        selected = self.uniforms[start : self.uniform_index]
        if self.chain_major:
            return selected[None, :].copy()
        return selected[:, None].copy()

    def uniform(self, low: float, high: float) -> float:
        value = self.uniforms[self.uniform_index]
        self.uniform_index += 1
        return float(low + value * (high - low))

    def choice(
        self,
        a: int,
        size: int,
        replace: bool = False,
    ) -> np.ndarray:
        del replace
        assert a == 1
        return np.zeros(size, dtype=int)


def _ar1_chain(n_steps: int = 2048, dimension: int = 3) -> np.ndarray:
    rng = np.random.default_rng(17)
    noise = rng.normal(size=(n_steps, dimension))
    chain = np.empty_like(noise)
    chain[0] = noise[0]
    for index in range(1, n_steps):
        chain[index] = 0.8 * chain[index - 1] + noise[index]
    return chain


def test_autocorrelation_is_scale_invariant() -> None:
    chain = _ar1_chain()
    lags = (1, 2, 4, 8, 16, 32)
    expected = autocorrelation_by_lag(chain, lags)

    for scale in (1e-12, 1e-8, 1e8, 1e12):
        actual = autocorrelation_by_lag(scale * chain, lags)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_whitened_thinning_is_scale_invariant() -> None:
    chain = _ar1_chain()
    kwargs = {
        "candidate_thinnings": (1, 2, 4, 8, 16, 32),
        "acf_threshold": 0.1,
        "stable_window": 2,
        "probe_mode": "both",
        "num_probes": 6,
        "whiten": True,
        "random_state": 9,
    }
    expected_thinning, expected_metadata = estimate_thinning_from_chain(chain, **kwargs)

    for scale in (1e-12, 1e12):
        thinning, metadata = estimate_thinning_from_chain(scale * chain, **kwargs)
        assert thinning == expected_thinning
        np.testing.assert_allclose(
            metadata["max_abs_autocorrelation"],
            expected_metadata["max_abs_autocorrelation"],
            rtol=2e-12,
            atol=2e-12,
        )


def test_constant_probe_has_zero_autocorrelation() -> None:
    series = np.ones((20, 2))
    np.testing.assert_array_equal(
        autocorrelation_by_lag(series, (1, 5)),
        np.zeros((2, 2)),
    )


def test_sample_covariance_eigendecomposition_reconstructs_covariance() -> None:
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(500, 3)) @ np.array(
        [
            [2.0, 0.5, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 0.4],
        ]
    )
    eigenvectors, eigenvalues = sample_covariance_eigendecomposition(samples)
    reconstructed = (eigenvectors * eigenvalues) @ eigenvectors.T

    np.testing.assert_allclose(
        reconstructed,
        np.cov(samples, rowvar=False),
        rtol=1e-12,
        atol=1e-12,
    )


def test_hit_and_run_accepts_covariance_shaped_directions() -> None:
    angle = np.deg2rad(35.0)
    eigenvectors = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    region = BoxDomain(np.array([[-2.0, 2.0], [-1.0, 1.0]])).as_region()
    sampler = HitAndRunSampler(
        burn_in=0,
        thinning=2,
        direction_eigenvectors=eigenvectors,
        direction_eigenvalues=np.array([4.0, 0.25]),
    )

    samples = sampler.sample(region, 100, random_state=3, x0=np.zeros(2))

    assert samples.shape == (100, 2)
    assert np.all(region.contains(samples))


@pytest.mark.parametrize(
    "mode",
    ["stepwise", "precomputed", "numba", "parallel_numba"],
)
def test_batched_hit_and_run_matches_scalar_sampler_with_fixed_stream(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    region = BoxDomain(np.array([[-1.5, 2.0], [-2.0, 1.0]])).as_region()
    x0 = np.array([0.25, -0.4])
    n_samples = 12
    burn_in = 3
    stream_rng = np.random.default_rng(29)
    directions = stream_rng.normal(size=(burn_in + n_samples, region.dimension))
    uniforms = stream_rng.random(size=burn_in + n_samples)

    scalar_stream = _FixedHitAndRunStream(directions, uniforms)
    monkeypatch.setattr(sampling_module, "as_rng", lambda _: scalar_stream)
    scalar_samples = HitAndRunSampler(
        burn_in=burn_in,
        thinning=1,
    ).sample(region, n_samples, x0=x0)

    batched_stream = _FixedHitAndRunStream(
        directions,
        uniforms,
        chain_major=mode == "parallel_numba",
    )
    monkeypatch.setattr(sampling_module, "as_rng", lambda _: batched_stream)
    sampler_kwargs = {"burn_in": burn_in, "thinning": 1, "n_chains": 1}
    if mode == "numba":
        sampler = NumbaBatchedHitAndRunSampler(**sampler_kwargs)
    elif mode == "parallel_numba":
        sampler = ParallelNumbaBatchedHitAndRunSampler(
            **sampler_kwargs,
            parallel_random=False,
        )
    else:
        sampler = BatchedHitAndRunSampler(
            **sampler_kwargs,
            precompute_ad=mode == "precomputed",
        )
    batched_samples = sampler.sample(region, n_samples, x0=x0)

    np.testing.assert_allclose(batched_samples, scalar_samples, rtol=1e-13, atol=1e-13)
    assert scalar_stream.direction_index == burn_in + n_samples
    assert scalar_stream.uniform_index == burn_in + n_samples
    assert batched_stream.direction_index == burn_in + n_samples
    assert batched_stream.uniform_index == burn_in + n_samples


def test_batched_implementations_match_for_multiple_chains() -> None:
    region = BoxDomain(np.array([[-1.5, 2.0], [-2.0, 1.0]])).as_region()
    starting_pool = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.5],
            [-0.75, 0.25],
            [1.0, 0.5],
            [-1.0, -1.0],
        ]
    )
    sampler_kwargs = {
        "burn_in": 2,
        "thinning": 3,
        "n_chains": 4,
    }

    stepwise = BatchedHitAndRunSampler(
        **sampler_kwargs,
        precompute_ad=False,
    ).sample(region, 11, random_state=37, x0=starting_pool)
    precomputed = BatchedHitAndRunSampler(
        **sampler_kwargs,
        precompute_ad=True,
    ).sample(region, 11, random_state=37, x0=starting_pool)
    compiled = NumbaBatchedHitAndRunSampler(**sampler_kwargs).sample(
        region,
        11,
        random_state=37,
        x0=starting_pool,
    )

    np.testing.assert_allclose(precomputed, stepwise, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(compiled, precomputed, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("parallel", [False, True])
def test_batched_sampler_records_requested_phase_timings(parallel: bool) -> None:
    region = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])).as_region()
    sampler: BatchedHitAndRunSampler
    if parallel:
        sampler = ParallelNumbaBatchedHitAndRunSampler(
            burn_in=0,
            thinning=2,
            n_chains=2,
            parallel_random=True,
            random_threads=2,
            profile_timing=True,
        )
    else:
        sampler = BatchedHitAndRunSampler(
            burn_in=0,
            thinning=2,
            n_chains=2,
            precompute_ad=True,
            profile_timing=True,
        )

    sampler.sample(
        region,
        6,
        random_state=43,
        x0=np.array([[0.0, 0.0], [0.25, -0.25]]),
    )

    assert set(sampler.last_phase_timings) == {
        "slack_matmul_seconds",
        "constraint_product_matmul_seconds",
        "random_generation_seconds",
        "transitions_seconds",
    }
    assert all(value >= 0.0 for value in sampler.last_phase_timings.values())


def test_numba_batched_sampler_matches_shaped_numpy_directions() -> None:
    angle = np.deg2rad(35.0)
    eigenvectors = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    region = BoxDomain(np.array([[-2.0, 2.0], [-1.0, 1.0]])).as_region()
    starting_pool = np.array(
        [[0.0, 0.0], [0.5, -0.5], [-0.75, 0.25], [1.0, 0.5]]
    )
    sampler_kwargs = {
        "burn_in": 2,
        "thinning": 3,
        "n_chains": 4,
        "direction_eigenvectors": eigenvectors,
        "direction_eigenvalues": np.array([4.0, 0.25]),
    }

    expected = BatchedHitAndRunSampler(
        **sampler_kwargs,
        precompute_ad=True,
    ).sample(region, 11, random_state=41, x0=starting_pool)
    actual = NumbaBatchedHitAndRunSampler(**sampler_kwargs).sample(
        region,
        11,
        random_state=41,
        x0=starting_pool,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.all(region.contains(actual, tol=1e-12))


def test_parallel_numba_kernel_matches_serial_numba_kernel() -> None:
    from art._numba_sampling import (
        sample_precomputed_chains,
        sample_precomputed_chains_parallel,
    )

    region = BoxDomain(np.array([[-1.5, 2.0], [-2.0, 1.0]])).as_region()
    points = np.array(
        [[0.0, 0.0], [0.5, -0.5], [-0.75, 0.25], [1.0, 0.5]]
    )
    slack = region.b[None, :] - points @ region.A.T
    burn_in = 2
    thinning = 3
    n_rounds = 3
    n_steps = burn_in + thinning * n_rounds
    rng = np.random.default_rng(53)
    directions = rng.normal(size=(n_steps, points.shape[0], region.dimension))
    uniforms = rng.random(size=(n_steps, points.shape[0]))
    constraint_products = (
        directions.reshape(-1, region.dimension) @ region.A.T
    ).reshape(n_steps, points.shape[0], -1)

    serial_points = points.copy()
    serial_slack = slack.copy()
    serial_samples = np.empty((n_rounds * points.shape[0], region.dimension))
    sample_precomputed_chains(
        serial_points,
        serial_slack,
        directions,
        constraint_products,
        uniforms,
        burn_in,
        thinning,
        serial_samples,
    )

    parallel_points = points.copy()
    parallel_slack = slack.copy()
    parallel_samples = np.empty_like(serial_samples)
    sample_precomputed_chains_parallel(
        parallel_points,
        parallel_slack,
        directions.transpose(1, 0, 2).copy(),
        constraint_products.transpose(1, 0, 2).copy(),
        uniforms.T.copy(),
        burn_in,
        thinning,
        parallel_samples,
    )

    np.testing.assert_allclose(parallel_samples, serial_samples, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(parallel_points, serial_points, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(parallel_slack, serial_slack, rtol=1e-13, atol=1e-13)


def test_parallel_numba_sampler_is_reproducible_with_partial_batch() -> None:
    region = BoxDomain(np.array([[-2.0, 2.0], [-1.0, 1.0]])).as_region()
    starting_pool = np.array(
        [[0.0, 0.0], [0.5, -0.5], [-0.75, 0.25], [1.0, 0.5]]
    )
    sampler = ParallelNumbaBatchedHitAndRunSampler(
        burn_in=2,
        thinning=3,
        n_chains=4,
    )

    first = sampler.sample(region, 11, random_state=59, x0=starting_pool)
    second = sampler.sample(region, 11, random_state=59, x0=starting_pool)

    np.testing.assert_array_equal(second, first)
    assert np.all(region.contains(first, tol=1e-12))


def test_parallel_numba_sampler_rejects_invalid_random_thread_count() -> None:
    region = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])).as_region()
    sampler = ParallelNumbaBatchedHitAndRunSampler(
        n_chains=2,
        random_threads=0,
    )

    with pytest.raises(ValueError, match="random_threads must be at least 1"):
        sampler.sample(region, 2, random_state=1, x0=np.zeros(2))


def test_parallel_numba_sampler_defaults() -> None:
    sampler = ParallelNumbaBatchedHitAndRunSampler()

    assert sampler.n_chains == 50
    assert sampler.parallel_random
    assert sampler.random_threads == 10


def test_hit_and_run_applies_trace_scaled_eigenvalue_floor() -> None:
    region = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])).as_region()
    sampler = HitAndRunSampler(
        burn_in=0,
        thinning=1,
        direction_eigenvectors=np.eye(2),
        direction_eigenvalues=np.array([1.0, 0.0]),
        direction_eigenvalue_floor=1e-4,
    )
    _, sqrt_eigenvalues = sampler._direction_spectrum(dimension=2)

    samples = sampler.sample(region, 20, random_state=1, x0=np.zeros(2))

    np.testing.assert_allclose(sqrt_eigenvalues, [1.0, np.sqrt(5e-5)])
    assert samples.shape == (20, 2)
    assert np.all(region.contains(samples))


def test_hit_and_run_step_interval_matches_box_boundaries() -> None:
    region = BoxDomain(np.array([[-1.0, 2.0], [-2.0, 4.0]])).as_region()
    sampler = HitAndRunSampler()

    t_low, t_high = sampler._step_interval(
        region,
        x=np.array([0.5, 1.0]),
        direction=np.array([0.6, 0.8]),
    )

    assert t_low == pytest.approx(-2.5)
    assert t_high == pytest.approx(2.5)


def test_hit_and_run_step_interval_checks_parallel_constraints() -> None:
    region = BoxDomain(np.array([[-1.0, 1.0], [-1.0, 1.0]])).as_region()
    sampler = HitAndRunSampler(feasibility_tol=1e-10)

    with pytest.raises(RuntimeError, match="Current point is infeasible"):
        sampler._step_interval(
            region,
            x=np.array([0.0, 1.0 + 2e-10]),
            direction=np.array([1.0, 0.0]),
        )


def test_trace_scaled_covariance_eigenvalue_floor() -> None:
    safe, floor = floor_covariance_eigenvalues(
        np.array([4.0, 0.0]),
        floor_ratio=1e-2,
    )

    assert floor == 2e-2
    np.testing.assert_allclose(safe, [4.0, 2e-2])
