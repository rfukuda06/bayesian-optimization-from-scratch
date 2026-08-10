import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel


@pytest.fixture
def data_1d():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(12, 1))
    y = np.sin(X).ravel() + 0.1 * rng.standard_normal(12)
    return X, y


def _fit_both(X, y, length_scale=1.5, signal_variance=2.0, noise_variance=0.01):
    ours = GaussianProcess(RBFKernel(length_scale, signal_variance), noise_variance)
    ours.fit(X, y)
    sk_kernel = ConstantKernel(signal_variance, "fixed") * RBF(length_scale, "fixed")
    theirs = GaussianProcessRegressor(
        kernel=sk_kernel, alpha=noise_variance, optimizer=None, normalize_y=False
    )
    theirs.fit(X, y)
    return ours, theirs


def test_mean_and_std_match_sklearn(data_1d):
    X, y = data_1d
    ours, theirs = _fit_both(X, y)
    X_star = np.linspace(-2, 12, 50)[:, None]
    mean, std = ours.predict(X_star)
    sk_mean, sk_std = theirs.predict(X_star, return_std=True)
    np.testing.assert_allclose(mean, sk_mean, atol=1e-6)
    np.testing.assert_allclose(std, sk_std, atol=1e-6)


def test_interpolates_training_data_when_noiseless(data_1d):
    X, y = data_1d
    # length_scale=1.0 keeps K well-conditioned (lambda_min ~ 1e-6), so the
    # 1e-10 Cholesky jitter floor does not visibly perturb interpolation;
    # at length_scale=1.5, lambda_min ~ 2e-9 and the jitter alone costs ~1e-3.
    gp = GaussianProcess(RBFKernel(1.0, 2.0), noise_variance=1e-10)
    gp.fit(X, y)
    mean, std = gp.predict(X)
    np.testing.assert_allclose(mean, y, atol=1e-4)
    assert std.max() < 1e-3


def test_reverts_to_prior_far_from_data(data_1d):
    X, y = data_1d
    gp = GaussianProcess(RBFKernel(1.5, 2.0), noise_variance=0.01)
    gp.fit(X, y)
    mean, std = gp.predict(np.array([[1000.0]]))
    np.testing.assert_allclose(mean, 0.0, atol=1e-6)          # zero-mean prior
    np.testing.assert_allclose(std, np.sqrt(2.0), atol=1e-6)  # sqrt(signal_variance)


def test_include_noise_adds_noise_variance(data_1d):
    X, y = data_1d
    gp = GaussianProcess(RBFKernel(1.5, 2.0), noise_variance=0.25)
    gp.fit(X, y)
    X_star = np.array([[5.0]])
    _, std_f = gp.predict(X_star)
    _, std_y = gp.predict(X_star, include_noise=True)
    np.testing.assert_allclose(std_y**2, std_f**2 + 0.25, atol=1e-10)


def test_predict_shapes(data_1d):
    X, y = data_1d
    gp = GaussianProcess(RBFKernel(1.5, 2.0), 0.01)
    gp.fit(X, y)
    mean, std = gp.predict(np.linspace(0, 10, 7))   # 1D input accepted
    assert mean.shape == (7,) and std.shape == (7,)
    mean, cov = gp.predict(np.linspace(0, 10, 7), return_cov=True)
    assert cov.shape == (7, 7)


def test_log_marginal_likelihood_matches_sklearn(data_1d):
    X, y = data_1d
    ours, theirs = _fit_both(X, y)
    np.testing.assert_allclose(
        ours.log_marginal_likelihood(),
        theirs.log_marginal_likelihood(),   # no args: LML at the fixed theta
        atol=1e-6,
    )


def test_fit_hyperparameters_improves_lml_within_bounds():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(25, 1))
    y = np.sin(6 * X).ravel() + 0.1 * rng.standard_normal(25)
    y = (y - y.mean()) / y.std()

    gp = GaussianProcess(RBFKernel(length_scale=5.0, signal_variance=0.1), 0.5)
    gp.fit(X, y)
    lml_before = gp.log_marginal_likelihood()

    gp.fit_hyperparameters(rng=np.random.default_rng(0))
    lml_after = gp.log_marginal_likelihood()

    assert lml_after >= lml_before - 1e-8
    from gpbo.gp import DEFAULT_HP_BOUNDS
    lo, hi = np.exp(DEFAULT_HP_BOUNDS[:, 0]), np.exp(DEFAULT_HP_BOUNDS[:, 1])
    theta = np.array(
        [gp.kernel.length_scale, gp.kernel.signal_variance, gp.noise_variance]
    )
    assert np.all(np.isfinite(theta))
    assert np.all(theta >= lo * (1 - 1e-9)) and np.all(theta <= hi * (1 + 1e-9))


def test_fit_hyperparameters_is_deterministic_given_rng():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(15, 1))
    y = np.sin(6 * X).ravel()

    results = []
    for _ in range(2):
        gp = GaussianProcess(RBFKernel(1.0, 1.0), 0.01)
        gp.fit(X, y)
        gp.fit_hyperparameters(rng=np.random.default_rng(3))
        results.append(
            (gp.kernel.length_scale, gp.kernel.signal_variance, gp.noise_variance)
        )
    assert results[0] == results[1]


def test_sample_posterior_shape_and_mean(data_1d):
    X, y = data_1d
    gp = GaussianProcess(RBFKernel(1.5, 2.0), 1e-8)
    gp.fit(X, y)
    X_star = np.linspace(0, 10, 30)[:, None]
    samples = gp.sample_posterior(X_star, 500, rng=np.random.default_rng(1))
    assert samples.shape == (500, 30)
    mean, _ = gp.predict(X_star)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.2)
