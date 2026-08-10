"""Expected Improvement acquisition function (maximization convention).

EI(x) = E[max(f(x) - y_best - xi, 0)] for f(x) ~ N(mu, sigma^2), which has
the closed form  EI = I*Phi(z) + sigma*phi(z),  I = mu - y_best - xi,
z = I/sigma. The first term is the exploitation part (probability-weighted
improvement of the mean), the second is the exploration part (reward for
uncertainty).
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(mean, std, y_best, xi=0.01) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    improvement = mean - y_best - xi
    # Where std ~ 0 the distribution is a point mass; z is undefined, EI is 0.
    safe = std > 1e-12
    z = np.where(safe, improvement / np.where(safe, std, 1.0), 0.0)
    ei = np.where(safe, improvement * norm.cdf(z) + std * norm.pdf(z), 0.0)
    return np.clip(ei, 0.0, None)
