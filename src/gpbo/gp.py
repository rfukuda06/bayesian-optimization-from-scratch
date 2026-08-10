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
from scipy.optimize import minimize

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

    def sample_posterior(self, X_star, n_samples, rng=None) -> np.ndarray:
        """Draw functions from the posterior: f = mu + L_c z, z ~ N(0, I),
        where L_c is the Cholesky factor of the posterior covariance."""
        rng = np.random.default_rng() if rng is None else rng
        mean, cov = self.predict(X_star, return_cov=True)
        m = len(mean)
        L_c = cholesky(cov + 1e-10 * np.eye(m), lower=True)
        z = rng.standard_normal((m, n_samples))
        return (mean[:, None] + L_c @ z).T

    def log_marginal_likelihood(self) -> float:
        """log p(y | X, theta) = -1/2 y^T alpha - sum(log L_ii) - n/2 log(2 pi).

        Term 1 rewards data fit, term 2 (which is 1/2 log|K|) penalizes model
        complexity — maximizing this trades the two off automatically.
        """
        if self._L is None:
            raise RuntimeError("Call fit() before log_marginal_likelihood().")
        n = len(self._y)
        return float(
            -0.5 * self._y @ self._alpha
            - np.sum(np.log(np.diag(self._L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def _get_log_theta(self):
        return np.log(
            [self.kernel.length_scale, self.kernel.signal_variance, self.noise_variance]
        )

    def _set_log_theta(self, log_theta):
        self.kernel.length_scale = float(np.exp(log_theta[0]))
        self.kernel.signal_variance = float(np.exp(log_theta[1]))
        self.noise_variance = float(np.exp(log_theta[2]))

    def fit_hyperparameters(self, bounds=None, n_restarts=5, rng=None) -> None:
        """Maximize the LML over theta = log(l, sf2, sn2) with multi-start L-BFGS-B.

        Optimizing in log space keeps parameters positive and puts the scales
        on comparable footing. The LML is multimodal, hence the restarts; the
        current theta is always kept as a candidate, so the fitted LML can
        never be worse than the starting one.
        """
        if self._L is None:
            raise RuntimeError("Call fit() before fit_hyperparameters().")
        if bounds is None:
            bounds = DEFAULT_HP_BOUNDS
        rng = np.random.default_rng() if rng is None else rng

        def neg_lml(log_theta):
            self._set_log_theta(log_theta)
            try:
                self._update_factorization()
                return -self.log_marginal_likelihood()
            except LinAlgError:
                return 1e10  # infeasible theta: huge penalty, finite for L-BFGS

        warm = self._get_log_theta()
        candidates = [(neg_lml(warm), warm)]  # baseline: keeping current theta
        starts = [warm] + [
            rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_restarts)
        ]
        for x0 in starts:
            res = minimize(neg_lml, x0, method="L-BFGS-B", bounds=bounds)
            if np.isfinite(res.fun):
                candidates.append((res.fun, res.x))

        best = min(candidates, key=lambda c: c[0])
        self._set_log_theta(best[1])
        self._update_factorization()
