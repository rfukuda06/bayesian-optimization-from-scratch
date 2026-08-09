"""Gaussian Process regression via Cholesky factorization.

Posterior equations (Rasmussen & Williams, ch. 2), for K = k(X,X) + sn2*I:
    alpha = K^-1 y          -> via cho_solve, never an explicit inverse
    mu*   = k(X*,X) alpha
    V     = L^-1 k(X,X*)
    var*  = k(x*,x*) - diag(V^T V)
The Cholesky route is O(n^3/3), numerically stable, and gives log|K| for
the marginal likelihood as 2*sum(log diag L).
"""

import warnings

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular

from gpbo.kernels import RBFKernel, _as_2d

# log-space bounds for (length_scale, signal_variance, noise_variance),
# stated in the spec for normalized-input / standardized-y data.
DEFAULT_HP_BOUNDS = np.log(np.array([[1e-2, 10.0], [1e-2, 1e2], [1e-8, 1.0]]))

_JITTERS = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6)


class GaussianProcess:
    def __init__(self, kernel: RBFKernel, noise_variance: float):
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self._X = None
        self._y = None
        self._L = None
        self._alpha = None

    def fit(self, X, y) -> None:
        self._X = _as_2d(X)
        self._y = np.asarray(y, dtype=float).ravel()
        self._update_factorization()

    def _update_factorization(self) -> None:
        n = len(self._y)
        K = self.kernel(self._X, self._X) + self.noise_variance * np.eye(n)
        for i, jitter in enumerate(_JITTERS):
            try:
                self._L = cholesky(K + jitter * np.eye(n), lower=True)
                break
            except LinAlgError:
                if i == len(_JITTERS) - 1:
                    raise
                warnings.warn(
                    f"Cholesky failed at jitter={jitter:.0e}; escalating."
                )
        self._alpha = cho_solve((self._L, True), self._y)

    def predict(self, X_star, return_cov=False, include_noise=False):
        if self._L is None:
            raise RuntimeError("Call fit() before predict().")
        X_star = _as_2d(X_star)
        K_s = self.kernel(self._X, X_star)          # (n, m)
        mean = K_s.T @ self._alpha                  # (m,)
        V = solve_triangular(self._L, K_s, lower=True)
        if return_cov:
            cov = self.kernel(X_star, X_star) - V.T @ V
            if include_noise:
                cov = cov + self.noise_variance * np.eye(len(X_star))
            return mean, cov
        # diag of k(X*,X*) is exactly signal_variance for the RBF kernel
        var = self.kernel.signal_variance - np.sum(V**2, axis=0)
        if include_noise:
            var = var + self.noise_variance
        var = np.maximum(var, 1e-12)  # cancellation can go slightly negative
        return mean, np.sqrt(var)
