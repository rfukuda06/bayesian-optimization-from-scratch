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


from gpbo.optimizer import BayesianOptimizer


def _quadratic(x):
    return -((x[0] - 1.2) ** 2)   # max at x = 1.2, value 0


def test_bo_finds_smooth_1d_maximum_within_budget():
    opt = BayesianOptimizer(_quadratic, bounds=[(0.0, 3.0)])
    result = opt.run(n_init=3, n_iter=12, seed=0)   # 15 evaluations total
    assert result.best_y >= -0.01
    assert abs(result.best_x[0] - 1.2) < 0.15


def test_bo_is_reproducible_for_same_seed():
    r1 = BayesianOptimizer(_quadratic, bounds=[(0.0, 3.0)]).run(3, 5, seed=42)
    r2 = BayesianOptimizer(_quadratic, bounds=[(0.0, 3.0)]).run(3, 5, seed=42)
    np.testing.assert_array_equal(r1.X, r2.X)
    np.testing.assert_array_equal(r1.y, r2.y)


def test_result_structure():
    result = BayesianOptimizer(_quadratic, bounds=[(0.0, 3.0)]).run(3, 4, seed=1)
    assert result.X.shape == (7, 1)
    assert result.y.shape == (7,)
    assert len(result.history) == 4
    assert np.all(np.diff(result.best_so_far) >= 0)        # running best
    assert np.all((result.X >= 0.0) & (result.X <= 3.0))   # original units
    rec = result.history[0]
    assert rec.X.shape == (3, 1) and rec.x_next.shape == (1,)
    assert len(rec.theta) == 3 and rec.ei_max >= 0
