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
