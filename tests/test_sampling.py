import numpy as np

from art.domain import BoxDomain
from art.sampling import (
    HitAndRunSampler,
    autocorrelation_by_lag,
    estimate_thinning_from_chain,
    floor_covariance_eigenvalues,
    sample_covariance_eigendecomposition,
)


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


def test_trace_scaled_covariance_eigenvalue_floor() -> None:
    safe, floor = floor_covariance_eigenvalues(
        np.array([4.0, 0.0]),
        floor_ratio=1e-2,
    )

    assert floor == 2e-2
    np.testing.assert_allclose(safe, [4.0, 2e-2])
