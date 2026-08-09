"""RBF (squared exponential) kernel.

k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))
"""

import numpy as np


def _as_2d(X):
    """Coerce input to float array of shape (n, d); 1D arrays become (n, 1)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


class RBFKernel:
    def __init__(self, length_scale: float, signal_variance: float):
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)

    def __call__(self, X1, X2) -> np.ndarray:
        X1, X2 = _as_2d(X1), _as_2d(X2)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, computed without an explicit
        # (n1, n2, d) intermediate; clipped at 0 because cancellation can
        # produce tiny negatives for near-identical points.
        sq = (
            np.sum(X1**2, axis=1)[:, None]
            + np.sum(X2**2, axis=1)[None, :]
            - 2.0 * X1 @ X2.T
        )
        sq = np.clip(sq, 0.0, None)
        return self.signal_variance * np.exp(-0.5 * sq / self.length_scale**2)
