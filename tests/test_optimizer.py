import numpy as np

from gpbo.optimizer import (
    _apply_duplicate_guard,
    _maximize_ei_candidates,
    _maximize_ei_grid,
)


class _StubGP:
    """Stands in for GaussianProcess: EI with constant std is maximized where
    the mean is maximized, so the acquisition argmax is known analytically."""

    def __init__(self, center):
        self.center = np.asarray(center, dtype=float)

    def predict(self, X, return_cov=False, include_noise=False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean = -np.sum((X - self.center) ** 2, axis=1)
        return mean, np.full(len(X), 0.3)


def test_grid_maximizer_finds_known_argmax_1d():
    x, ei = _maximize_ei_grid(_StubGP([0.7]), y_best=-1.0)
    assert abs(x[0] - 0.7) < 0.01
    assert ei > 0


def test_candidate_maximizer_finds_known_argmax_2d():
    x, ei = _maximize_ei_candidates(
        _StubGP([0.3, 0.6]), y_best=-1.0, rng=np.random.default_rng(0), d=2
    )
    assert np.linalg.norm(x - np.array([0.3, 0.6])) < 0.02
    assert ei > 0


def test_duplicate_guard_replaces_repeat_point():
    rng = np.random.default_rng(0)
    X = np.array([[0.5, 0.5], [0.2, 0.8]])
    guarded = _apply_duplicate_guard(np.array([0.5, 0.5]), X, rng)
    assert np.linalg.norm(guarded - np.array([0.5, 0.5])) > 1e-6


def test_duplicate_guard_passes_through_new_point():
    rng = np.random.default_rng(0)
    X = np.array([[0.5, 0.5]])
    x = np.array([0.1, 0.9])
    np.testing.assert_array_equal(_apply_duplicate_guard(x, X, rng), x)
