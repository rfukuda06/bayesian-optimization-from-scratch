"""Bayesian optimization loop and acquisition maximization.

All functions here operate in the unit box [0,1]^d with standardized y;
BayesianOptimizer owns the mapping to and from original units.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

from gpbo.acquisition import expected_improvement
from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel


def _maximize_ei_grid(gp, y_best, n_grid=1000):
    """1D: dense grid — also exactly what the visualizations plot."""
    grid = np.linspace(0.0, 1.0, n_grid)[:, None]
    mean, std = gp.predict(grid)
    ei = expected_improvement(mean, std, y_best)
    i = int(np.argmax(ei))
    return grid[i], float(ei[i])


def _maximize_ei_candidates(gp, y_best, rng, d, n_candidates=2048, n_refine=5):
    """d >= 2: seeded random candidates, then L-BFGS-B refinement of the top few.

    The dimension d must come from the caller (the GP has no notion of it)."""
    candidates = rng.uniform(size=(n_candidates, d))
    mean, std = gp.predict(candidates)
    ei = expected_improvement(mean, std, y_best)

    def neg_ei(x):
        m, s = gp.predict(x[None, :])
        return -expected_improvement(m, s, y_best)[0]

    best_x, best_ei = candidates[int(np.argmax(ei))], float(ei.max())
    for i in np.argsort(ei)[-n_refine:]:
        res = minimize(
            neg_ei, candidates[i], method="L-BFGS-B", bounds=[(0.0, 1.0)] * d
        )
        if -res.fun > best_ei:
            best_x, best_ei = res.x, float(-res.fun)
    return best_x, best_ei


def _apply_duplicate_guard(x, X_existing, rng):
    """If EI collapsed onto an already-sampled point, fall back to random
    so the loop cannot stall re-evaluating the same location.

    Late in a converged 1D run this fires often (the grid argmax keeps
    landing on the incumbent), so late proposals can look random-uniform —
    that is expected behavior, not a bug."""
    if np.min(np.linalg.norm(X_existing - x, axis=1)) < 1e-6:
        return rng.uniform(size=x.shape[0])
    return x


@dataclass
class IterationRecord:
    X: np.ndarray          # observations so far, original units, (n_i, d)
    y: np.ndarray          # (n_i,)
    theta: tuple           # fitted (length_scale, signal_variance, noise_variance)
    x_next: np.ndarray     # point evaluated next (post-guard), original units, (d,)
    ei_max: float          # EI at the pre-guard argmax; when the duplicate
                           # guard fired, x_next is a random fallback and this
                           # EI describes the replaced proposal, not x_next


@dataclass(repr=False)   # custom __repr__: the default would dump the arrays
class OptimizationResult:
    X: np.ndarray          # all evaluated points, original units, (n, d)
    y: np.ndarray          # (n,)
    best_x: np.ndarray
    best_y: float
    best_so_far: np.ndarray
    history: list          # list[IterationRecord], one per BO iteration

    def __repr__(self) -> str:
        n, d = self.X.shape
        return (
            f"OptimizationResult(n={n}, d={d}, best_y={self.best_y:.4f}; "
            f"arrays: X, y, best_so_far; history: {len(self.history)} iterations)"
        )


class BayesianOptimizer:
    """Maximizes `objective` over box `bounds` ((d, 2), original units).

    Owns all data conditioning: inputs live in [0,1]^d internally, y is
    standardized to zero mean / unit variance before each GP fit (the GP
    prior has zero mean — raw values far from 0 would distort EI).
    """

    def __init__(self, objective, bounds):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)

    def _to_orig(self, x_unit):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + x_unit * (hi - lo)

    def run(self, n_init, n_iter, seed) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        d = len(self.bounds)

        X_unit = rng.uniform(size=(n_init, d))
        y = np.array([self.objective(self._to_orig(x)) for x in X_unit])

        # theta persists across iterations on this object -> warm starts.
        gp = GaussianProcess(RBFKernel(length_scale=0.3, signal_variance=1.0), 1e-4)
        history = []

        for _ in range(n_iter):
            y_mean, y_std = y.mean(), y.std()
            y_std = y_std if y_std > 1e-12 else 1.0
            y_s = (y - y_mean) / y_std

            gp.fit(X_unit, y_s)
            gp.fit_hyperparameters(rng=rng)

            y_best = y_s.max()
            if d == 1:
                x_next, ei_max = _maximize_ei_grid(gp, y_best)
            else:
                x_next, ei_max = _maximize_ei_candidates(gp, y_best, rng, d=d)
            x_next = _apply_duplicate_guard(np.asarray(x_next).ravel(), X_unit, rng)

            history.append(
                IterationRecord(
                    X=self._to_orig(X_unit).copy(),
                    y=y.copy(),
                    theta=(
                        gp.kernel.length_scale,
                        gp.kernel.signal_variance,
                        gp.noise_variance,
                    ),
                    x_next=self._to_orig(x_next),
                    ei_max=ei_max,
                )
            )

            y_new = self.objective(self._to_orig(x_next))
            X_unit = np.vstack([X_unit, x_next])
            y = np.append(y, y_new)

        best = int(np.argmax(y))
        return OptimizationResult(
            X=self._to_orig(X_unit),
            y=y,
            best_x=self._to_orig(X_unit[best]),
            best_y=float(y[best]),
            best_so_far=np.maximum.accumulate(y),
            history=history,
        )
