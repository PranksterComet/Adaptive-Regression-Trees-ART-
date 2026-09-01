from __future__ import annotations

import numpy as np
import pytest

from art.domain import BoxDomain
from examples.benchmark_functions import (
    GAUSSIAN_DEFAULT_INTERVAL,
    PLANE_WAVE_DEFAULT_INTERVAL,
    QUADRATIC_DEFAULT_INTERVAL,
    RASTRIGIN_DEFAULT_INTERVAL,
    ROSENBROCK_DEFAULT_INTERVAL,
    SPHERICAL_PIECEWISE_DEFAULT_INTERVAL,
    GaussianFunction,
    GaussianMixtureFunction,
    PlaneWaveFunction,
    QuadraticFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphericalPiecewisePolynomialFunction,
    default_gaussian_mixture_2d,
    rotation_matrix_2d,
    sphere_radius_for_box_volume_fraction,
)
from examples.tree_2D_benchmark import add_output_offset


def test_rotation_matrix_2d_supports_radians_and_degrees() -> None:
    radians = rotation_matrix_2d(np.pi / 2.0)
    degrees = rotation_matrix_2d(90.0, degrees=True)

    assert np.allclose(radians, degrees)
    assert np.allclose(radians @ np.array([1.0, 0.0]), [0.0, 1.0])
    assert np.allclose(radians.T @ radians, np.eye(2))
    assert np.isclose(np.linalg.det(radians), 1.0)


def test_paper_default_domains() -> None:
    quadratic = BoxDomain.hypercube(2, *QUADRATIC_DEFAULT_INTERVAL)
    gaussian = BoxDomain.hypercube(2, *GAUSSIAN_DEFAULT_INTERVAL)
    rosenbrock = BoxDomain.hypercube(2, *ROSENBROCK_DEFAULT_INTERVAL)
    plane_wave = BoxDomain.hypercube(2, *PLANE_WAVE_DEFAULT_INTERVAL)
    spherical = BoxDomain.hypercube(2, *SPHERICAL_PIECEWISE_DEFAULT_INTERVAL)
    rastrigin = BoxDomain.hypercube(2, *RASTRIGIN_DEFAULT_INTERVAL)

    assert np.array_equal(quadratic.bounds, [[-3.0, 3.0], [-3.0, 3.0]])
    assert np.array_equal(gaussian.bounds, [[-3.0, 3.0], [-3.0, 3.0]])
    assert np.array_equal(rosenbrock.bounds, [[-2.0, 3.0], [-2.0, 3.0]])
    assert np.array_equal(plane_wave.bounds, quadratic.bounds)
    assert np.array_equal(spherical.bounds, quadratic.bounds)
    assert np.array_equal(rastrigin.bounds, [[-5.12, 5.12], [-5.12, 5.12]])


def test_quadratic_paper_defaults_and_batch_evaluation() -> None:
    function = QuadraticFunction(dimension=2)
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    expected_second = 1.0 + 3.0 + 1.0 + 4.0 / 8.0

    assert np.array_equal(function.Q, np.eye(2))
    assert np.allclose(np.diag(function.Lambda), [1.0, 1.0 / 8.0])
    assert function(X[0]) == 1.0
    assert np.allclose(function(X), [1.0, expected_second])


def test_quadratic_accepts_explicit_parameters() -> None:
    Q = np.array([[0.0, -1.0], [1.0, 0.0]])
    function = QuadraticFunction(
        dimension=2,
        beta=2.0,
        w=np.array([3.0, -1.0]),
        Q=Q,
        Lambda=np.array([2.0, 4.0]),
    )
    x = np.array([1.0, 2.0])
    matrix = Q @ np.diag([2.0, 4.0]) @ Q.T
    expected = 2.0 + np.array([3.0, -1.0]) @ x + x @ matrix @ x

    assert np.isclose(function(x), expected)


def test_output_offset_shifts_scalar_and_batch_benchmarks() -> None:
    base = PlaneWaveFunction(dimension=2, beta=0.0)
    shifted = add_output_offset(base, 2.5)
    X = np.array([[0.0, 0.0], [1.0, -0.5]])

    assert np.allclose(shifted(X), base(X) + 2.5)
    assert np.isclose(shifted(X[0]), base(X[0]) + 2.5)


def test_spherical_piecewise_polynomial_routes_scalar_and_batch_inputs() -> None:
    inside = QuadraticFunction(
        dimension=3,
        beta=1.0,
        w=np.array([1.0, 0.0, 0.0]),
        Lambda=np.zeros(3),
    )
    outside = QuadraticFunction(
        dimension=3,
        beta=-2.0,
        w=np.zeros(3),
        Lambda=np.array([1.0, 0.0, 0.0]),
    )
    function = SphericalPiecewisePolynomialFunction(
        inside_polynomial=inside,
        outside_polynomial=outside,
        radius=1.0,
        center=np.array([0.5, 0.0, 0.0]),
    )
    X = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    assert function.dimension == 3
    assert np.allclose(function(X), [1.5, 2.5, 2.0])
    assert np.isclose(function(X[0]), 1.5)


def test_spherical_piecewise_polynomial_validates_pieces_and_radius() -> None:
    piece_2d = QuadraticFunction(dimension=2)
    piece_3d = QuadraticFunction(dimension=3)

    with np.testing.assert_raises_regex(ValueError, "same dimension"):
        SphericalPiecewisePolynomialFunction(piece_2d, piece_3d, radius=1.0)
    with np.testing.assert_raises_regex(ValueError, "radius must be positive"):
        SphericalPiecewisePolynomialFunction(piece_2d, piece_2d, radius=0.0)


def test_sphere_radius_matches_requested_box_volume_fraction() -> None:
    bounds = np.array([[-2.0, 2.0], [-1.0, 3.0], [-3.0, 1.0]])
    center = np.mean(bounds, axis=1)
    radius = sphere_radius_for_box_volume_fraction(
        bounds,
        volume_fraction=0.3,
        center=center,
        n_probe=50_000,
        random_state=7,
    )
    rng = np.random.default_rng(19)
    points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50_000, 3))
    measured = np.mean(np.linalg.norm(points - center, axis=1) <= radius)

    assert measured == pytest.approx(0.3, abs=0.01)
    assert radius == sphere_radius_for_box_volume_fraction(
        bounds,
        volume_fraction=0.3,
        center=center,
        n_probe=50_000,
        random_state=7,
    )


def test_plane_wave_supports_general_dimension_and_normalizes_normal() -> None:
    function = PlaneWaveFunction(
        dimension=3,
        beta=2.0,
        amplitude=3.0,
        frequency=0.5,
        normal=np.array([3.0, 4.0, 0.0]),
        phase=np.pi / 2.0,
    )
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]])
    expected = 2.0 + 3.0 * np.cos(
        0.5 * (X @ np.array([3.0 / 5.0, 4.0 / 5.0, 0.0])) + np.pi / 2.0
    )

    assert np.isclose(np.linalg.norm(function.normal), 1.0)
    assert np.allclose(function(X), expected)
    assert np.isclose(function(X[0]), expected[0])


def test_plane_wave_random_features_are_reproducible() -> None:
    first = PlaneWaveFunction(
        dimension=5,
        feature_mode="random",
        random_state=11,
    )
    second = PlaneWaveFunction(
        dimension=5,
        feature_mode="random",
        random_state=11,
    )

    assert np.allclose(first.normal, second.normal)
    assert np.isclose(first.phase, second.phase)
    assert np.isclose(np.linalg.norm(first.normal), 1.0)
    assert 0.0 <= first.phase < 2.0 * np.pi


def test_gaussian_caches_precision_and_supports_random_rotation() -> None:
    first = GaussianFunction(dimension=3, q_mode="random", random_state=7)
    second = GaussianFunction(dimension=3, q_mode="random", random_state=7)
    precision_id = id(first.precision)
    X = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])

    values = first(X)

    assert np.allclose(first.Q.T @ first.Q, np.eye(3))
    assert np.allclose(first.Q, second.Q)
    assert np.isclose(values[0], 2.0)
    assert values[1] > 1.0
    assert values[1] < 2.0
    assert id(first.precision) == precision_id


def test_gaussian_accepts_full_sigma() -> None:
    Sigma = np.array([[2.0, 0.25], [0.25, 1.0]])
    function = GaussianFunction(dimension=2, beta=0.5, Sigma=Sigma)
    x = np.array([0.4, -0.2])
    expected = 0.5 + np.exp(-0.5 * (x @ np.linalg.inv(Sigma) @ x))

    assert np.allclose(function.covariance, Sigma)
    assert np.isclose(function(x), expected)


def test_shifted_gaussian_and_mixture_evaluation() -> None:
    first = GaussianFunction(
        dimension=2,
        beta=0.0,
        Sigma=np.array([0.5, 0.25]),
        mean=np.array([-1.0, 0.5]),
    )
    second = GaussianFunction(
        dimension=2,
        beta=0.0,
        Sigma=np.array([0.2, 0.8]),
        mean=np.array([1.0, -0.5]),
    )
    mixture = GaussianMixtureFunction(
        components=(first, second),
        weights=np.array([1.0, 3.0]),
        beta=1.0,
    )
    X = np.array([[-1.0, 0.5], [0.0, 0.0]])
    expected = 1.0 + 0.25 * first(X) + 0.75 * second(X)

    assert np.isclose(first(first.mean), 1.0)
    assert np.allclose(mixture(X), expected)
    assert np.isclose(mixture(X[0]), expected[0])


def test_default_gaussian_mixture_has_four_varied_components_in_domain() -> None:
    mixture = default_gaussian_mixture_2d()
    low, high = GAUSSIAN_DEFAULT_INTERVAL
    centers = np.vstack([component.mean for component in mixture.components])
    precisions = np.stack([component.precision for component in mixture.components])

    assert mixture.dimension == 2
    assert len(mixture.components) == 4
    assert mixture.beta == 1.0
    assert np.isclose(np.sum(mixture.weights), 1.0)
    assert np.unique(mixture.weights).size > 1
    assert np.all((centers >= low) & (centers <= high))
    assert not np.allclose(precisions, precisions[0])
    assert np.all(np.isfinite(mixture(centers)))


def test_default_gaussian_mixture_accepts_custom_beta() -> None:
    default = default_gaussian_mixture_2d()
    zero_baseline = default_gaussian_mixture_2d(beta=0.0)
    X = np.array([[0.2, -0.4], [1.0, 1.5]])

    assert default.beta == 1.0
    assert zero_baseline.beta == 0.0
    assert np.allclose(default(X) - zero_baseline(X), 1.0)


def test_rosenbrock_defaults_and_custom_coefficients() -> None:
    default = RosenbrockFunction(dimension=3)
    custom = RosenbrockFunction(dimension=2, a=2.0, b=5.0)
    X = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    assert np.allclose(default(X), [0.0, 2.0])
    assert custom(np.array([1.0, 3.0])) == 21.0


def test_rastrigin_standard_parameters_and_batch_evaluation() -> None:
    function = RastriginFunction(dimension=3)
    custom = RastriginFunction(dimension=2, A=2.0)
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    assert function.A == 10.0
    assert np.allclose(function(X), [0.0, 3.0, 60.75])
    assert function(X[0]) == 0.0
    assert np.isclose(custom(np.array([0.5, 0.0])), 4.25)
