import numpy as np
import pytest

from gpbo.kernels import RBFKernel


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_known_value():
    # k([0], [2]) with l=1, sf2=1 is exp(-|0-2|^2 / 2) = exp(-2)
    k = RBFKernel(length_scale=1.0, signal_variance=1.0)
    K = k(np.array([0.0]), np.array([2.0]))
    assert K.shape == (1, 1)
    np.testing.assert_allclose(K[0, 0], np.exp(-2.0), rtol=1e-12)


def test_diagonal_equals_signal_variance(rng):
    k = RBFKernel(length_scale=0.7, signal_variance=2.5)
    X = rng.uniform(size=(6, 2))
    K = k(X, X)
    np.testing.assert_allclose(np.diag(K), 2.5, rtol=1e-12)


def test_symmetry(rng):
    k = RBFKernel(length_scale=1.3, signal_variance=1.0)
    X = rng.uniform(size=(8, 3))
    K = k(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_positive_semidefinite(rng):
    k = RBFKernel(length_scale=0.5, signal_variance=1.0)
    X = rng.uniform(size=(20, 2))
    eigvals = np.linalg.eigvalsh(k(X, X))
    assert eigvals.min() >= -1e-8


def test_monotonic_decay():
    k = RBFKernel(length_scale=1.0, signal_variance=1.0)
    x0 = np.array([[0.0]])
    distances = np.array([[0.0], [0.5], [1.0], [2.0], [4.0]])
    values = k(x0, distances).ravel()
    assert np.all(np.diff(values) < 0)


def test_shapes_1d_and_2d_inputs(rng):
    k = RBFKernel(length_scale=1.0, signal_variance=1.0)
    assert k(rng.uniform(size=5), rng.uniform(size=3)).shape == (5, 3)      # 1D arrays
    assert k(rng.uniform(size=(5, 2)), rng.uniform(size=(3, 2))).shape == (5, 3)
