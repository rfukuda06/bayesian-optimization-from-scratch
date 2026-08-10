import numpy as np

from gpbo.acquisition import expected_improvement


def test_ei_nonnegative_everywhere():
    rng = np.random.default_rng(0)
    ei = expected_improvement(
        rng.standard_normal(200), rng.uniform(0, 2, 200), y_best=0.5
    )
    assert np.all(ei >= 0)


def test_ei_zero_when_std_zero():
    ei = expected_improvement(
        np.array([0.0, 5.0]), np.array([0.0, 0.0]), y_best=1.0
    )
    np.testing.assert_array_equal(ei, [0.0, 0.0])


def test_ei_grows_with_std_at_fixed_mean():
    stds = np.array([0.1, 0.5, 1.0, 2.0])
    ei = expected_improvement(np.zeros(4), stds, y_best=1.0)
    assert np.all(np.diff(ei) > 0)


def test_ei_matches_monte_carlo():
    # EI(x) = E[max(f - y_best - xi, 0)], f ~ N(mu, sigma^2).
    # Fixed seed + 1e5 draws per spec: deterministic, rtol ~1e-2.
    rng = np.random.default_rng(123)
    cases = [(0.0, 1.0, 0.0), (1.0, 0.5, 1.2), (-1.0, 2.0, 0.5)]
    for mu, sigma, y_best in cases:
        draws = rng.normal(mu, sigma, size=100_000)
        mc = np.maximum(draws - y_best - 0.01, 0.0).mean()
        closed = expected_improvement(
            np.array([mu]), np.array([sigma]), y_best
        )[0]
        np.testing.assert_allclose(closed, mc, rtol=2e-2, atol=5e-3)
