# GP Regression + Bayesian Optimization from Scratch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Gaussian Process regression and Bayesian optimization from scratch (kernel, Cholesky posterior, log-marginal-likelihood hyperparameter learning, Expected Improvement, BO loop), validated against sklearn, demonstrated on synthetic functions and a real SVM hyperparameter-tuning experiment vs. random search.

**Architecture:** An installable package `src/gpbo` with four focused modules (`kernels`, `gp`, `acquisition`, `optimizer`), consumed by three experiment scripts and four test files. `GaussianProcess` is pure math on the arrays it receives; `BayesianOptimizer` owns all data conditioning (unit-box inputs, standardized y, maximize-always convention). One seeded `np.random.default_rng` is threaded through every stochastic step so identical seeds give bit-identical runs.

**Tech Stack:** Python ≥3.11, uv, numpy, scipy, matplotlib, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-08-09-gp-bayesian-optimization-design.md` — the authority on all math. Code comments should be derivation-grade (state the equation being computed and why the stable form is used).

**Milestone pushes:** `git push` steps appear at the end of Tasks 6, 9, 11, 12, and 14.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `src/gpbo/__init__.py` | Public re-exports |
| `src/gpbo/kernels.py` | `RBFKernel` — covariance matrices only |
| `src/gpbo/gp.py` | `GaussianProcess` — fit/predict/LML/hyperparameter fitting/posterior sampling |
| `src/gpbo/acquisition.py` | `expected_improvement` — pure array function |
| `src/gpbo/optimizer.py` | `BayesianOptimizer`, `OptimizationResult`, `IterationRecord`, EI maximization, duplicate guard |
| `experiments/gp_demo.py` | 1D GP figure, sklearn comparison, recovery sanity print |
| `experiments/synthetic_optimization.py` | 1D BO frames + 2D Branin figures |
| `experiments/hyperparameter_tuning.py` | SVM-on-digits: BO vs random search, 10 seeds |
| `tests/test_kernels.py`, `tests/test_gp.py`, `tests/test_acquisition.py`, `tests/test_optimizer.py` | Test suite per spec §7 |
| `docs/math-walkthrough.md` | Derivations (Task 13) |
| `README.md` | Project story, figures, results (Task 14) |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/gpbo/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "gpbo"
version = "0.1.0"
description = "Gaussian Process regression and Bayesian optimization from scratch"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "scikit-learn>=1.4",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gpbo"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package**

`src/gpbo/__init__.py`:

```python
"""Gaussian Process regression and Bayesian optimization, from scratch."""

from gpbo.kernels import RBFKernel
from gpbo.gp import GaussianProcess
from gpbo.acquisition import expected_improvement
from gpbo.optimizer import BayesianOptimizer, OptimizationResult

__all__ = [
    "RBFKernel",
    "GaussianProcess",
    "expected_improvement",
    "BayesianOptimizer",
    "OptimizationResult",
]
```

NOTE: this import list only resolves after Task 9. Until then, keep `__init__.py` EMPTY (just the docstring) and add the imports as their modules land: `RBFKernel` in Task 2, `GaussianProcess` in Task 3, `expected_improvement` in Task 7, optimizer names in Task 9. Each task's commit updates it.

- [ ] **Step 3: Install and verify**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`, installs gpbo editable plus all deps.

Run: `uv run pytest`
Expected: `no tests ran` (exit code 5 — fine, nothing collected yet).

Run: `uv run python -c "import gpbo; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/gpbo/__init__.py
git commit -m "chore: scaffold gpbo package with uv"
```

---

### Task 2: RBF kernel

**Files:**
- Create: `src/gpbo/kernels.py`
- Test: `tests/test_kernels.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_kernels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kernels.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'gpbo.kernels'`

- [ ] **Step 3: Implement the kernel**

`src/gpbo/kernels.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_kernels.py -v`
Expected: 6 passed

- [ ] **Step 5: Export and commit**

Add to `src/gpbo/__init__.py`: `from gpbo.kernels import RBFKernel` (and `"RBFKernel"` to `__all__`).

```bash
git add src/gpbo/kernels.py src/gpbo/__init__.py tests/test_kernels.py
git commit -m "feat: RBF kernel with covariance-matrix construction"
```

---

### Task 3: GaussianProcess fit + predict (Cholesky posterior)

**Files:**
- Create: `src/gpbo/gp.py`
- Test: `tests/test_gp.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_gp.py`:

```python
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
    gp = GaussianProcess(RBFKernel(1.5, 2.0), noise_variance=1e-10)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'gpbo.gp'`

- [ ] **Step 3: Implement fit + predict**

`src/gpbo/gp.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp.py tests/test_kernels.py -v`
Expected: 11 passed

- [ ] **Step 5: Export and commit**

Add to `src/gpbo/__init__.py`: `from gpbo.gp import GaussianProcess` (and `"GaussianProcess"` to `__all__`).

```bash
git add src/gpbo/gp.py src/gpbo/__init__.py tests/test_gp.py
git commit -m "feat: GP posterior via Cholesky, validated against sklearn"
```

---

### Task 4: Log marginal likelihood

**Files:**
- Modify: `src/gpbo/gp.py` (add one method to `GaussianProcess`)
- Test: `tests/test_gp.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_gp.py`)

```python
def test_log_marginal_likelihood_matches_sklearn(data_1d):
    X, y = data_1d
    ours, theirs = _fit_both(X, y)
    np.testing.assert_allclose(
        ours.log_marginal_likelihood(),
        theirs.log_marginal_likelihood(),   # no args: LML at the fixed theta
        atol=1e-6,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gp.py::test_log_marginal_likelihood_matches_sklearn -v`
Expected: FAIL with `AttributeError: ... has no attribute 'log_marginal_likelihood'`

- [ ] **Step 3: Implement** (add method to `GaussianProcess` in `src/gpbo/gp.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/gp.py tests/test_gp.py
git commit -m "feat: log marginal likelihood, matches sklearn's value"
```

---

### Task 5: Hyperparameter fitting by LML maximization

**Files:**
- Modify: `src/gpbo/gp.py` (add `fit_hyperparameters`)
- Test: `tests/test_gp.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_gp.py`)

Per the spec, this test is deterministic: it asserts LML improvement and bounded, finite parameters — NOT parameter recovery (recovery is flaky by nature and lives as an informational print in `gp_demo.py`).

```python
def test_fit_hyperparameters_improves_lml_within_bounds():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(25, 1))
    y = np.sin(6 * X).ravel() + 0.1 * rng.standard_normal(25)
    y = (y - y.mean()) / y.std()

    gp = GaussianProcess(RBFKernel(length_scale=5.0, signal_variance=0.1), 0.5)
    gp.fit(X, y)
    lml_before = gp.log_marginal_likelihood()

    gp.fit_hyperparameters(rng=np.random.default_rng(0))
    lml_after = gp.log_marginal_likelihood()

    assert lml_after >= lml_before - 1e-8
    from gpbo.gp import DEFAULT_HP_BOUNDS
    lo, hi = np.exp(DEFAULT_HP_BOUNDS[:, 0]), np.exp(DEFAULT_HP_BOUNDS[:, 1])
    theta = np.array(
        [gp.kernel.length_scale, gp.kernel.signal_variance, gp.noise_variance]
    )
    assert np.all(np.isfinite(theta))
    assert np.all(theta >= lo * (1 - 1e-9)) and np.all(theta <= hi * (1 + 1e-9))


def test_fit_hyperparameters_is_deterministic_given_rng():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(15, 1))
    y = np.sin(6 * X).ravel()

    results = []
    for _ in range(2):
        gp = GaussianProcess(RBFKernel(1.0, 1.0), 0.01)
        gp.fit(X, y)
        gp.fit_hyperparameters(rng=np.random.default_rng(3))
        results.append(
            (gp.kernel.length_scale, gp.kernel.signal_variance, gp.noise_variance)
        )
    assert results[0] == results[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp.py -k fit_hyperparameters -v`
Expected: 2 FAIL with `AttributeError: ... 'fit_hyperparameters'`

- [ ] **Step 3: Implement** (add to `GaussianProcess` in `src/gpbo/gp.py`; also add `from scipy.optimize import minimize` to the imports)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/gp.py tests/test_gp.py
git commit -m "feat: hyperparameter learning by multi-start LML maximization"
```

---

### Task 6: Posterior sampling

**Files:**
- Modify: `src/gpbo/gp.py` (add `sample_posterior`)
- Test: `tests/test_gp.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_gp.py`)

```python
def test_sample_posterior_shape_and_mean(data_1d):
    X, y = data_1d
    gp = GaussianProcess(RBFKernel(1.5, 2.0), 1e-8)
    gp.fit(X, y)
    X_star = np.linspace(0, 10, 30)[:, None]
    samples = gp.sample_posterior(X_star, 500, rng=np.random.default_rng(1))
    assert samples.shape == (500, 30)
    mean, _ = gp.predict(X_star)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gp.py::test_sample_posterior_shape_and_mean -v`
Expected: FAIL with `AttributeError: ... 'sample_posterior'`

- [ ] **Step 3: Implement** (add to `GaussianProcess`)

```python
    def sample_posterior(self, X_star, n_samples, rng=None) -> np.ndarray:
        """Draw functions from the posterior: f = mu + L_c z, z ~ N(0, I),
        where L_c is the Cholesky factor of the posterior covariance."""
        rng = np.random.default_rng() if rng is None else rng
        mean, cov = self.predict(X_star, return_cov=True)
        m = len(mean)
        L_c = cholesky(cov + 1e-10 * np.eye(m), lower=True)
        z = rng.standard_normal((m, n_samples))
        return (mean[:, None] + L_c @ z).T
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: 15 passed (6 kernel + 9 gp)

- [ ] **Step 5: Commit and push (milestone: core GP complete)**

```bash
git add src/gpbo/gp.py tests/test_gp.py
git commit -m "feat: posterior function sampling"
git push
```

---

### Task 7: Expected Improvement

**Files:**
- Create: `src/gpbo/acquisition.py`
- Test: `tests/test_acquisition.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_acquisition.py`:

```python
import numpy as np

from gpbo.acquisition import expected_improvement


def test_ei_nonnegative_everywhere():
    rng = np.random.default_rng(0)
    ei = expected_improvement(
        rng.standard_normal(200), rng.uniform(0, 2, 200), y_best=0.5
    )
    assert np.all(ei >= 0)


def test_ei_zero_when_std_zero():
    ei = expected_improvement(
        np.array([0.0, 5.0]), np.array([0.0, 0.0]), y_best=1.0
    )
    np.testing.assert_array_equal(ei, [0.0, 0.0])


def test_ei_grows_with_std_at_fixed_mean():
    stds = np.array([0.1, 0.5, 1.0, 2.0])
    ei = expected_improvement(np.zeros(4), stds, y_best=1.0)
    assert np.all(np.diff(ei) > 0)


def test_ei_matches_monte_carlo():
    # EI(x) = E[max(f - y_best - xi, 0)], f ~ N(mu, sigma^2).
    # Fixed seed + 1e5 draws per spec: deterministic, rtol ~1e-2.
    rng = np.random.default_rng(123)
    cases = [(0.0, 1.0, 0.0), (1.0, 0.5, 1.2), (-1.0, 2.0, 0.5)]
    for mu, sigma, y_best in cases:
        draws = rng.normal(mu, sigma, size=100_000)
        mc = np.maximum(draws - y_best - 0.01, 0.0).mean()
        closed = expected_improvement(
            np.array([mu]), np.array([sigma]), y_best
        )[0]
        np.testing.assert_allclose(closed, mc, rtol=2e-2, atol=5e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_acquisition.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'gpbo.acquisition'`

- [ ] **Step 3: Implement**

`src/gpbo/acquisition.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquisition.py -v`
Expected: 4 passed

- [ ] **Step 5: Export and commit**

Add to `src/gpbo/__init__.py`: `from gpbo.acquisition import expected_improvement` (and `"expected_improvement"` to `__all__`).

```bash
git add src/gpbo/acquisition.py src/gpbo/__init__.py tests/test_acquisition.py
git commit -m "feat: Expected Improvement with Monte-Carlo-validated closed form"
```

---

### Task 8: Acquisition maximization + duplicate guard

**Files:**
- Create: `src/gpbo/optimizer.py` (helpers only in this task; the optimizer class lands in Task 9)
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_optimizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'gpbo.optimizer'`

- [ ] **Step 3: Implement the helpers**

`src/gpbo/optimizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/optimizer.py tests/test_optimizer.py
git commit -m "feat: EI maximization (grid + refined candidates) and duplicate guard"
```

---

### Task 9: BayesianOptimizer loop

**Files:**
- Modify: `src/gpbo/optimizer.py` (add dataclasses + `BayesianOptimizer`)
- Test: `tests/test_optimizer.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_optimizer.py`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: collection ERROR with `ImportError: cannot import name 'BayesianOptimizer'` — the import at the top of the file fails, so none of the file's tests run until Step 3.

- [ ] **Step 3: Implement** (append to `src/gpbo/optimizer.py`; extend imports with `from dataclasses import dataclass`, `from gpbo.gp import GaussianProcess`, `from gpbo.kernels import RBFKernel`)

```python
@dataclass
class IterationRecord:
    X: np.ndarray          # observations so far, original units, (n_i, d)
    y: np.ndarray          # (n_i,)
    theta: tuple           # fitted (length_scale, signal_variance, noise_variance)
    x_next: np.ndarray     # proposed point, original units, (d,)
    ei_max: float          # EI value at the (pre-guard) proposal


@dataclass
class OptimizationResult:
    X: np.ndarray          # all evaluated points, original units, (n, d)
    y: np.ndarray          # (n,)
    best_x: np.ndarray
    best_y: float
    best_so_far: np.ndarray
    history: list          # list[IterationRecord], one per BO iteration


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
```

NOTE: `self._to_orig(X_unit)` broadcasts over rows because `lo`/`hi` have shape `(d,)` — it works for both a single point `(d,)` and a batch `(n, d)`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: 26 passed (6 kernel + 9 gp + 4 acquisition + 7 optimizer)

- [ ] **Step 5: Export, commit, push (milestone: library complete)**

Finalize `src/gpbo/__init__.py` to the full version shown in Task 1 Step 2 (adding `BayesianOptimizer` and `OptimizationResult`).

```bash
git add src/gpbo/optimizer.py src/gpbo/__init__.py tests/test_optimizer.py
git commit -m "feat: Bayesian optimization loop with conditioning and history"
git push
```

---

### Task 10: Experiment — 1D GP demo

**Files:**
- Create: `experiments/gp_demo.py`
- Output: `figures/gp_demo.png`

- [ ] **Step 1: Write the experiment**

`experiments/gp_demo.py`:

```python
"""1D GP visualization: prior samples, posterior with 3 then 8 observations.

Also prints (a) the sklearn agreement check at fixed hyperparameters,
(b) our fitted LML vs sklearn's fitted LML, and (c) the non-blocking
hyperparameter-recovery sanity experiment from the spec.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

NOISE_STD = 0.2


def f(x):
    return x * np.sin(x)


def plot_panel(ax, gp, X_obs, y_obs, title, rng):
    xs = np.linspace(0, 10, 400)[:, None]
    mean, std = gp.predict(xs)
    ax.plot(xs, f(xs.ravel()), "k--", lw=1, label="true f(x) = x sin(x)")
    ax.fill_between(
        xs.ravel(), mean - 2 * std, mean + 2 * std, alpha=0.2, label="±2σ"
    )
    ax.plot(xs, mean, lw=2, label="posterior mean")
    for s in gp.sample_posterior(xs, 4, rng=rng):
        ax.plot(xs, s, lw=0.6, alpha=0.6)
    ax.plot(X_obs, y_obs, "ko", ms=6, label="observations")
    ax.set_title(title)
    ax.set_xlim(0, 10)


def main():
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    # (a) Prior: zero mean, bands at ±2 sqrt(signal_variance), sampled functions
    # drawn directly from the kernel matrix — no fit involved.
    xs = np.linspace(0, 10, 400)[:, None]
    prior_kernel = RBFKernel(length_scale=1.0, signal_variance=4.0)
    K = prior_kernel(xs, xs) + 1e-10 * np.eye(len(xs))
    L = np.linalg.cholesky(K)
    axes[0].plot(xs, L @ rng.standard_normal((len(xs), 4)), lw=0.8)
    axes[0].fill_between(xs.ravel(), -2 * 2.0, 2 * 2.0, alpha=0.15)
    axes[0].axhline(0, color="C0", lw=2)
    axes[0].set_title("(a) GP prior: mean 0, ±2σ_f, 4 sampled functions")

    # (b), (c) Posterior after 3 and 8 noisy observations, hyperparameters fitted.
    X_all = rng.uniform(0.5, 9.5, size=(8, 1))
    y_all = f(X_all.ravel()) + NOISE_STD * rng.standard_normal(8)
    for ax, n, label in [(axes[1], 3, "(b)"), (axes[2], 8, "(c)")]:
        gp = GaussianProcess(RBFKernel(1.0, 4.0), NOISE_STD**2)
        gp.fit(X_all[:n], y_all[:n])
        gp.fit_hyperparameters(rng=np.random.default_rng(1))
        plot_panel(
            ax, gp, X_all[:n], y_all[:n], f"{label} posterior, {n} observations", rng
        )
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "gp_demo.png", dpi=150)
    print(f"saved {FIGDIR / 'gp_demo.png'}")

    # --- sklearn agreement at fixed theta -------------------------------------
    ours = GaussianProcess(RBFKernel(1.5, 4.0), 0.04)
    ours.fit(X_all, y_all)
    sk = GaussianProcessRegressor(
        kernel=ConstantKernel(4.0, "fixed") * RBF(1.5, "fixed"),
        alpha=0.04,
        optimizer=None,
        normalize_y=False,
    ).fit(X_all, y_all)
    xs_test = np.linspace(0, 10, 100)[:, None]
    m1, s1 = ours.predict(xs_test)
    m2, s2 = sk.predict(xs_test, return_std=True)
    print(f"fixed-theta max|Δmean| = {np.abs(m1 - m2).max():.2e}")
    print(f"fixed-theta max|Δstd|  = {np.abs(s1 - s2).max():.2e}")

    # --- fitted LML: ours vs sklearn's optimizer ------------------------------
    fitted = GaussianProcess(RBFKernel(1.0, 1.0), 0.1)
    fitted.fit(X_all, y_all)
    fitted.fit_hyperparameters(rng=np.random.default_rng(2))
    sk_opt = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0), alpha=0.04, n_restarts_optimizer=5
    ).fit(X_all, y_all)
    print(f"our fitted LML     = {fitted.log_marginal_likelihood():.4f}")
    print(f"sklearn fitted LML = {sk_opt.log_marginal_likelihood():.4f}")
    print("(not directly comparable models: sklearn's noise here is fixed alpha)")

    # --- non-blocking recovery sanity experiment (spec §6.1 / §7) -------------
    true_l = 1.0
    gen = RBFKernel(true_l, 1.0)
    Xg = np.linspace(0, 10, 40)[:, None]
    Kg = gen(Xg, Xg) + 1e-8 * np.eye(40)
    yg = np.linalg.cholesky(Kg) @ np.random.default_rng(5).standard_normal(40)
    rec = GaussianProcess(RBFKernel(3.0, 0.5), 1e-4)
    rec.fit(Xg, yg)
    rec.fit_hyperparameters(rng=np.random.default_rng(6))
    print(
        f"recovery sanity: true length_scale={true_l}, "
        f"recovered={rec.kernel.length_scale:.3f} (informational only)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python experiments/gp_demo.py`
Expected: prints the saved path; `fixed-theta max|Δmean|` and `max|Δstd|` both ≤ 1e-6; recovery prints a length scale within roughly [0.5, 2.0]; `figures/gp_demo.png` exists.

- [ ] **Step 3: Inspect the figure**

Open `figures/gp_demo.png`. Check: prior panel shows wiggly zero-centered samples; posterior bands are tight at observations, wide between/beyond them; the band in (c) is visibly tighter than (b).

- [ ] **Step 4: Commit**

```bash
git add experiments/gp_demo.py figures/gp_demo.png
git commit -m "feat: 1D GP demo with sklearn comparison and recovery sanity check"
```

---

### Task 11: Experiment — synthetic optimization (1D frames + Branin)

**Files:**
- Create: `experiments/synthetic_optimization.py`
- Output: `figures/bo_1d_iter_*.png`, `figures/bo_branin_samples.png`, `figures/bo_branin_gp_mean.png`, `figures/bo_branin_regret.png`

- [ ] **Step 1: Write the experiment**

`experiments/synthetic_optimization.py`:

```python
"""BO on synthetic functions, replotted from OptimizationResult.history.

1D: f(x) = -sin(3x) - x^2 + 0.7x on [-1, 2] — per-iteration frames.
2D: Branin (negated for the maximize convention) — samples, GP mean, regret.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpbo.acquisition import expected_improvement
from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel
from gpbo.optimizer import BayesianOptimizer

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)


def gp_from_record(rec, bounds):
    """Rebuild the exact GP of one BO iteration from its history record:
    same observations, same fitted theta, same conditioning as the optimizer."""
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    X_unit = (rec.X - lo) / (hi - lo)
    y_std = rec.y.std()
    y_std = y_std if y_std > 1e-12 else 1.0
    y_s = (rec.y - rec.y.mean()) / y_std
    l, sf2, sn2 = rec.theta
    gp = GaussianProcess(RBFKernel(l, sf2), sn2)
    gp.fit(X_unit, y_s)
    return gp, rec.y.mean(), y_std, y_s.max()


# ---------------------------------------------------------------- 1D problem
def f1d(x):
    return -np.sin(3 * x[0]) - x[0] ** 2 + 0.7 * x[0]


def run_1d():
    bounds = [(-1.0, 2.0)]
    result = BayesianOptimizer(f1d, bounds).run(n_init=3, n_iter=12, seed=3)

    xs = np.linspace(-1, 2, 400)
    xs_unit = ((xs - (-1.0)) / 3.0)[:, None]
    f_true = np.array([f1d([x]) for x in xs])

    for i, rec in enumerate(result.history, start=1):
        gp, y_mean, y_std, y_best_s = gp_from_record(rec, bounds)
        mean_s, std_s = gp.predict(xs_unit)
        mean = y_mean + y_std * mean_s          # back to original units
        band = 2 * y_std * std_s
        ei = expected_improvement(mean_s, std_s, y_best_s)

        fig, (top, bot) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1]
        )
        top.plot(xs, f_true, "k--", lw=1, label="true objective")
        top.fill_between(xs, mean - band, mean + band, alpha=0.2, label="±2σ")
        top.plot(xs, mean, lw=2, label="GP mean")
        top.plot(rec.X, rec.y, "ko", ms=6, label="samples")
        top.axvline(rec.x_next[0], color="r", ls=":", label="next sample")
        top.set_title(f"Bayesian optimization, iteration {i}")
        top.legend(loc="lower left", fontsize=7)
        bot.plot(xs, ei, color="g")
        bot.axvline(rec.x_next[0], color="r", ls=":")
        bot.set_ylabel("EI (standardized)")
        bot.set_xlabel("x")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"bo_1d_iter_{i:02d}.png", dpi=120)
        plt.close(fig)

    print(f"1D: best f = {result.best_y:.4f} at x = {result.best_x[0]:.4f}")
    print(f"saved {len(result.history)} frames to figures/bo_1d_iter_*.png")


# ------------------------------------------------------------- Branin (2D)
BRANIN_MIN = 0.397887


def branin(x):
    x1, x2 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def run_branin():
    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    result = BayesianOptimizer(lambda x: -branin(x), bounds).run(
        n_init=5, n_iter=25, seed=0
    )

    g1, g2 = np.meshgrid(np.linspace(-5, 10, 120), np.linspace(0, 15, 120))
    Z = np.array(
        [branin([a, b]) for a, b in zip(g1.ravel(), g2.ravel())]
    ).reshape(g1.shape)

    # Samples over the true landscape (log-spaced levels: Branin spans decades)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cs = ax.contourf(g1, g2, Z, levels=np.logspace(-0.5, 2.5, 20), cmap="viridis")
    fig.colorbar(cs)
    ax.plot(result.X[:5, 0], result.X[:5, 1], "ws", ms=7, label="initial")
    ax.plot(result.X[5:, 0], result.X[5:, 1], "wo", ms=5, label="BO samples")
    for i, (x1, x2) in enumerate(result.X[5:], start=1):
        ax.annotate(str(i), (x1, x2), color="w", fontsize=6)
    ax.plot(*result.best_x, "r*", ms=15, label="best found")
    ax.set_title("Branin: sample sequence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_samples.png", dpi=150)
    plt.close(fig)

    # Final GP mean surface (negate back to minimization view for display)
    rec = result.history[-1]
    gp, y_mean, y_std, _ = gp_from_record(rec, bounds)
    U = np.column_stack(
        [(g1.ravel() + 5.0) / 15.0, g2.ravel() / 15.0]
    )
    mean_s, _ = gp.predict(U)
    surrogate = -(y_mean + y_std * mean_s).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cs = ax.contourf(g1, g2, surrogate, levels=20, cmap="viridis")
    fig.colorbar(cs)
    ax.plot(result.X[:, 0], result.X[:, 1], "wo", ms=4)
    ax.set_title("Final GP mean (surrogate of Branin)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_gp_mean.png", dpi=150)
    plt.close(fig)

    # Regret curve. best_so_far is the running max of -branin, so negating it
    # gives the running min of branin itself — already monotone.
    regret = -result.best_so_far - BRANIN_MIN
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(regret) + 1), np.maximum(regret, 1e-6))
    ax.set_xlabel("evaluation")
    ax.set_ylabel("|best − f*|")
    ax.set_title("Branin: simple regret")
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_regret.png", dpi=150)
    plt.close(fig)

    print(f"Branin: best value {-result.best_y:.4f} (global min {BRANIN_MIN})")


if __name__ == "__main__":
    run_1d()
    run_branin()
```

- [ ] **Step 2: Run it**

Run: `uv run python experiments/synthetic_optimization.py`
Expected (≈1–3 min): prints 1D best near x ≈ −0.36 / f ≈ 0.50 (the global maximum of −sin 3x − x² + 0.7x; a lower local maximum sits near x ≈ 1.57); Branin best value within ~0.5 of 0.397887; 12 + 3 figures in `figures/`.

- [ ] **Step 3: Inspect figures**

`bo_1d_iter_01.png` → wide bands, EI spread out; last frame → samples clustered near the global max, EI nearly flat. `bo_branin_samples.png` → later samples concentrated near one of the three Branin minima.

- [ ] **Step 4: Commit and push (milestone: synthetic BO demonstrated)**

```bash
git add experiments/synthetic_optimization.py figures/bo_1d_iter_*.png figures/bo_branin_*.png
git commit -m "feat: synthetic BO experiments with per-iteration visualization"
git push
```

---

### Task 12: Experiment — SVM hyperparameter tuning vs random search

**Files:**
- Create: `experiments/hyperparameter_tuning.py`
- Output: `figures/hp_comparison.png`, `figures/hp_landscape.png`, `data/digits_landscape.npz`

- [ ] **Step 1: Write the experiment**

`experiments/hyperparameter_tuning.py`:

```python
"""BO vs random search tuning SVC(C, gamma) on digits (spec §6.3).

Search space: a = log10(C) in [-3, 3], b = log10(gamma) in [-5, 1].
Objective: mean 5-fold stratified CV accuracy on the 80% pool.
Budget: 25 evaluations per method per trial; 10 seeded trials each.
Held-out 20% test set is touched exactly once per method at the end.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

from gpbo.optimizer import BayesianOptimizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGDIR, DATADIR = ROOT / "figures", ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)

BOUNDS = [(-3.0, 3.0), (-5.0, 1.0)]
N_SEEDS = 10
N_INIT, N_ITER = 5, 20          # 25 evaluations total
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

digits = load_digits()
X_pool, X_test, y_pool, y_test = train_test_split(
    digits.data / 16.0, digits.target, test_size=0.2, stratify=digits.target,
    random_state=0,
)


def objective(params):
    a, b = params
    clf = SVC(C=10.0**a, gamma=10.0**b)
    return cross_val_score(clf, X_pool, y_pool, cv=CV, n_jobs=-1).mean()


def random_search(n_evals, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    params = lo + rng.uniform(size=(n_evals, 2)) * (hi - lo)
    y = np.array([objective(p) for p in params])
    return params, y


def landscape():
    """20x20 ground-truth CV-accuracy grid, cached (~10 min first run)."""
    cache = DATADIR / "digits_landscape.npz"
    if cache.exists():
        d = np.load(cache)
        return d["A"], d["B"], d["Z"]
    A, B = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-5, 1, 20))
    Z = np.array(
        [objective([a, b]) for a, b in zip(A.ravel(), B.ravel())]
    ).reshape(A.shape)
    np.savez(cache, A=A, B=B, Z=Z)
    return A, B, Z


def main():
    bo_curves, rs_curves = [], []
    bo_best, rs_best = (-np.inf, None), (-np.inf, None)
    for seed in range(N_SEEDS):
        r = BayesianOptimizer(objective, BOUNDS).run(N_INIT, N_ITER, seed=seed)
        if seed == 0:
            r0 = r  # kept for the landscape plot below
        bo_curves.append(r.best_so_far)
        if r.best_y > bo_best[0]:
            bo_best = (r.best_y, r.best_x)

        params, y = random_search(N_INIT + N_ITER, seed=100 + seed)
        rs_curves.append(np.maximum.accumulate(y))
        if y.max() > rs_best[0]:
            rs_best = (y.max(), params[np.argmax(y)])
        print(f"seed {seed}: BO best {bo_curves[-1][-1]:.4f}  "
              f"RS best {rs_curves[-1][-1]:.4f}")

    bo_curves, rs_curves = np.array(bo_curves), np.array(rs_curves)
    evals = np.arange(1, N_INIT + N_ITER + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curves, label, color in [
        (bo_curves, "Bayesian optimization", "C0"),
        (rs_curves, "Random search", "C1"),
    ]:
        m, s = curves.mean(axis=0), curves.std(axis=0)
        ax.plot(evals, m, color=color, lw=2, label=label)
        ax.fill_between(evals, m - s, m + s, color=color, alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best 5-fold CV accuracy so far")
    ax.set_title(f"SVC(C, γ) on digits — mean ± std over {N_SEEDS} seeds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "hp_comparison.png", dpi=150)
    plt.close(fig)

    for n in (5, 10, 25):
        print(f"mean best after {n:2d} evals:  "
              f"BO {bo_curves[:, n - 1].mean():.4f}   "
              f"RS {rs_curves[:, n - 1].mean():.4f}")

    # BO sample placement of the seed-0 run (saved above) over the landscape
    A, B, Z = landscape()
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(A, B, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, label="CV accuracy")
    ax.plot(r0.X[:N_INIT, 0], r0.X[:N_INIT, 1], "ws", ms=7, label="initial")
    ax.plot(r0.X[N_INIT:, 0], r0.X[N_INIT:, 1], "wo", ms=5, label="BO samples")
    ax.plot(*r0.best_x, "r*", ms=15, label="best")
    ax.set_xlabel("log10 C")
    ax.set_ylabel("log10 gamma")
    ax.set_title("Where BO samples (seed 0)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hp_landscape.png", dpi=150)
    plt.close(fig)

    # Held-out test: one shot per method with its overall best config
    for name, (_, best_params) in [("BO", bo_best), ("RS", rs_best)]:
        a, b = best_params
        clf = SVC(C=10.0**a, gamma=10.0**b).fit(X_pool, y_pool)
        print(f"{name} best config: C=10^{a:.2f}, gamma=10^{b:.2f}  "
              f"-> held-out test accuracy {clf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python experiments/hyperparameter_tuning.py`
Expected (~15–35 min total on first run, including the cached landscape): per-seed lines print; mean best after 25 evals ≥ ~0.97 for both methods with BO ahead at 5–10 evals; two figures plus `data/digits_landscape.npz` created; held-out test accuracies within ~0.01 of the CV numbers.

NOTE: this step is long-running. Run it in the background and continue only when it finishes.

- [ ] **Step 3: Record the printed numbers**

Copy the `mean best after N evals` table and both held-out test accuracies into a scratch note — Task 14's README results table needs them verbatim.

- [ ] **Step 4: Commit and push (milestone: real ML experiment done)**

```bash
git add experiments/hyperparameter_tuning.py figures/hp_comparison.png figures/hp_landscape.png data/digits_landscape.npz
git commit -m "feat: SVM hyperparameter tuning — BO vs random search over 10 seeds"
git push
```

---

### Task 13: Math walkthrough document

**Files:**
- Create: `docs/math-walkthrough.md`

- [ ] **Step 1: Write the document**

Write `docs/math-walkthrough.md` with exactly these sections. Every derivation must be worked, not stated — the target reader is Renzo rehearsing explanations; each section ends by pointing at the implementing code (file + function).

1. **From Gaussians to function priors.** Univariate → multivariate Gaussian; covariance matrix meaning; a GP as an infinite-dimensional Gaussian where any finite slice of function values is jointly Gaussian with covariance given by the kernel. Show why RBF's length scale controls wiggliness and σ_f² controls amplitude. → `kernels.py`.
2. **Conditioning: the posterior equations.** Start from the joint prior over train/test values with observation noise,
   `[y; f*] ~ N(0, [[K + σn²I, K*], [K*ᵀ, K**]])`,
   apply the Gaussian conditioning identity, and derive
   `μ* = K*ᵀ(K + σn²I)⁻¹y` and `Σ* = K** − K*ᵀ(K + σn²I)⁻¹K*`. State the conditioning identity being used. → `gp.py: predict`.
3. **Why Cholesky.** Condition number squaring under explicit inversion; solve-vs-invert; `α = L⁻ᵀL⁻¹y`; how `log|K| = 2Σ log Lᵢᵢ` falls out for free; jitter and why it is principled (equivalent to a tiny extra noise term). → `gp.py: _update_factorization`.
4. **The log marginal likelihood.** Derive `log p(y|X,θ)` from the Gaussian density; identify the data-fit term, the complexity penalty, and the constant; explain the automatic Occam's razor with a sketch of how too-small and too-large length scales each lose. Include the analytic gradient formula `∂LML/∂θⱼ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θⱼ)` as the noted-but-unimplemented extension. → `gp.py: log_marginal_likelihood, fit_hyperparameters`.
5. **Expected Improvement, derived.** `EI(x) = E[max(f − y_best − ξ, 0)]`; work the truncated-Gaussian integral to `I·Φ(z) + σ·φ(z)`; interpret the two terms as exploitation and exploration; the σ→0 limit; the noisy-objective caveat (y_best is the best *noisy* observation — slightly optimistic; noise-robust variants exist and are out of scope). → `acquisition.py`.
6. **The BO loop and its conventions.** Why maximize-only; why inputs are normalized to the unit box (one shared length scale must be meaningful); why y is standardized (zero-mean prior honesty); the duplicate guard. → `optimizer.py`.
7. **Limitations.** O(n³) scaling and its practical ceiling; curse of dimensionality for BO; LML multimodality.

- [ ] **Step 2: Verify internal references**

Every `→ file: function` pointer must name a function that exists. Check against the actual code.

- [ ] **Step 3: Commit**

```bash
git add docs/math-walkthrough.md
git commit -m "docs: math walkthrough — derivations behind every module"
```

---

### Task 14: README + final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README** with these sections:

1. **Title + one-paragraph pitch** — GP regression and Bayesian optimization from scratch, validated against sklearn, applied to real hyperparameter tuning.
2. **The story in one diagram** — the outline's pipeline (Gaussians → kernels → prior → posterior → EI → BO → tuning → comparison).
3. **Results** — embed `figures/gp_demo.png`, one early + one late `bo_1d_iter_*.png`, `figures/bo_branin_samples.png`, `figures/hp_comparison.png`, `figures/hp_landscape.png`. Include the results table from Task 12 Step 3 (mean best after 5/10/25 evals for BO and RS) and the held-out test accuracies, with one honest paragraph: BO wins early; random search largely catches up by 25 evaluations on this well-bounded 2D space (cite Bergstra & Bengio) — the win is evaluation efficiency.
4. **What is implemented from scratch** — bullet list: RBF kernel, Cholesky posterior, LML + multi-start fitting, EI closed form, BO loop; and what is delegated to scipy/numpy (linear algebra, L-BFGS-B, norm pdf/cdf).
5. **Correctness** — how the sklearn agreement tests and the EI Monte-Carlo test work; `uv run pytest` badge-style line.
6. **How to run** — `uv sync`, then the three experiment commands with expected runtimes.
7. **Limitations** — O(n³), 1–3D scope, noisy-EI caveat.
8. **References** — Rasmussen & Williams (GPML, ch. 2 & 5); Bergstra & Bengio (2012); the spec and walkthrough docs by relative link.

- [ ] **Step 2: Full-suite verification**

Run: `uv run pytest -v`
Expected: 26 passed, no warnings other than possible matplotlib font cache notes.

Run: `git status`
Expected: only `README.md` untracked/modified; working tree otherwise clean.

- [ ] **Step 3: Commit and push (milestone: project complete)**

```bash
git add README.md
git commit -m "docs: README with results, figures, and honest comparison"
git push
```

---

## Deferred (spec: optional follow-ons — do NOT implement)

Matérn-2.5 kernel, UCB acquisition, GIF animations of BO iterations, analytic LML gradients. Listed here so nobody "helpfully" adds them.
