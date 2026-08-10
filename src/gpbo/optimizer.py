"""Bayesian optimization loop and acquisition maximization.

All functions here operate in the unit box [0,1]^d with standardized y;
BayesianOptimizer (Task 9) owns the mapping to and from original units.
"""

import numpy as np
from scipy.optimize import minimize

from gpbo.acquisition import expected_improvement


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
    so the loop cannot stall re-evaluating the same location."""
    if np.min(np.linalg.norm(X_existing - x, axis=1)) < 1e-6:
        return rng.uniform(size=x.shape[0])
    return x
