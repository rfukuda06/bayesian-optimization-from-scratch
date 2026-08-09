# Gaussian Process Regression + Bayesian Optimization from Scratch — Design Spec

- **Date:** 2026-08-09
- **Status:** Approved section-by-section in brainstorming; pending final user review of this document
- **Author:** Renzo Fukuda, with Claude (design dialogue)

## 0. Decisions log

| Decision | Choice |
|---|---|
| Kernel hyperparameters | Learned from scratch by maximizing the log marginal likelihood (scipy L-BFGS-B, multi-start) |
| Real ML experiment | RBF-kernel SVM on sklearn's bundled digits dataset, tuning log₁₀C and log₁₀γ |
| Implementation role | Claude writes all code with derivation-grade annotation, plus a math walkthrough document |
| Scope tier | Core-complete: outline + robustness fixes + multi-seed comparison + full tests + walkthrough; stretch items listed as optional follow-ons |
| Environment | Python 3.13, uv-managed, `pyproject.toml`; deps: numpy, scipy, matplotlib, scikit-learn; dev: pytest |

No external resources are required: all datasets ship inside scikit-learn, and every dependency is a plain package install. No accounts, downloads, API keys, or GPU.

## 1. Purpose and success criteria

Build Gaussian Process regression and Bayesian optimization largely from scratch to understand and implement the underlying mathematics, not to wrap an existing library. NumPy/SciPy are used for numerical linear algebra (matmul, Cholesky, triangular solves, L-BFGS-B, norm pdf/cdf); everything conceptually "the algorithm" — kernel, posterior, log marginal likelihood, Expected Improvement, the BO loop — is implemented here.

The project succeeds when:

1. The from-scratch GP posterior matches `sklearn.gaussian_process.GaussianProcessRegressor` to ~1e-6 under identical fixed hyperparameters.
2. The from-scratch log marginal likelihood matches sklearn's `log_marginal_likelihood()` at identical hyperparameters.
3. The BO loop, with hyperparameters learned by marginal likelihood, finds the optimum of synthetic benchmarks in few evaluations, with per-iteration visualizations of posterior, uncertainty, and acquisition.
4. The SVM-on-digits experiment produces a fair, multi-seed comparison of Bayesian optimization vs. random search under an equal evaluation budget.
5. Renzo can explain every concept on the outline's list by walking through the code and the math walkthrough document.

## 2. Scope

**In scope:** GP regression (RBF kernel only), zero-mean prior, Cholesky-based posterior, log-marginal-likelihood hyperparameter learning, Expected Improvement, the BO loop, three experiments (1D GP demo, synthetic 1D + 2D Branin optimization, SVM hyperparameter tuning vs. random search), full test suite, math walkthrough, README with committed figures.

**Out of scope** (per the outline): GP classification, sparse/approximate GPs, high-dimensional BO (everything here is 1–3D), multi-objective optimization, additional kernels, dashboards, distributed/production infrastructure, reimplementing linear algebra.

**Optional follow-ons** (explicitly not required; add only if the core is finished): Matérn-2.5 kernel, UCB acquisition for comparison, animated GIFs of BO iterations, analytic LML gradients.

## 3. Architecture

```text
bayesian-optimization-from-scratch/
├── src/gpbo/                      # installable package (pyproject.toml, uv)
│   ├── __init__.py
│   ├── kernels.py                 # RBFKernel
│   ├── gp.py                      # GaussianProcess
│   ├── acquisition.py             # expected_improvement
│   └── optimizer.py               # BayesianOptimizer, OptimizationResult
├── experiments/
│   ├── gp_demo.py                 # 1D GP visualization + sklearn comparison
│   ├── synthetic_optimization.py  # BO on 1D demo function + 2D Branin
│   └── hyperparameter_tuning.py   # SVM-on-digits: BO vs random search
├── tests/
│   ├── test_kernels.py
│   ├── test_gp.py
│   ├── test_acquisition.py
│   └── test_optimizer.py
├── docs/                          # this spec, implementation plan, math-walkthrough.md
├── figures/                       # committed output images (embedded in README)
├── data/                          # cached ground-truth accuracy grid for digits
├── pyproject.toml
└── README.md
```

`src/gpbo` is a real package installed editable (`uv sync`), so experiments and tests import `from gpbo.gp import GaussianProcess` with no path hacks, and everything runs via `uv run pytest` / `uv run experiments/gp_demo.py`.

### Interfaces

```python
# kernels.py
class RBFKernel:
    def __init__(self, length_scale: float, signal_variance: float): ...
    def __call__(self, X1, X2) -> np.ndarray:   # covariance matrix, shape (n1, n2)
        # k(x, x') = signal_variance * exp(-||x - x'||² / (2 * length_scale²))

# gp.py
class GaussianProcess:
    def __init__(self, kernel: RBFKernel, noise_variance: float): ...
    def fit(self, X, y) -> None                          # Cholesky factorization happens here
    def predict(self, X_star, return_cov=False, include_noise=False) -> (mean, std)  # or (mean, cov)
    def log_marginal_likelihood(self) -> float
    def fit_hyperparameters(self, bounds=None, n_restarts=5, rng=None) -> None
    def sample_posterior(self, X_star, n_samples, rng=None) -> np.ndarray

# acquisition.py
def expected_improvement(mean, std, y_best, xi=0.01) -> np.ndarray   # pure array function

# optimizer.py
class BayesianOptimizer:
    def __init__(self, objective, bounds): ...           # bounds: (d, 2) array in original units
    def run(self, n_init, n_iter, seed) -> OptimizationResult

@dataclass
class OptimizationResult:
    X, y                      # all evaluated points/values, original units
    best_x, best_y
    best_so_far               # per-evaluation running best
    history                   # per iteration: observations so far, fitted θ, x_next, max EI
```

### Separation of concerns

`GaussianProcess` is pure math on the arrays it is given. `BayesianOptimizer` owns all practical conditioning: mapping inputs to the unit box via the bounds, standardizing y to zero mean/unit variance, the maximize-always convention, and un-mapping results for reporting. This keeps each unit independently testable.

### Reproducibility rule

`BayesianOptimizer.run(seed)` creates a single `np.random.default_rng(seed)` and threads it through everything stochastic: initial design, candidate sampling, restart draws in `fit_hyperparameters`. Same seed → bit-identical run.

## 4. Mathematical specification

### Conventions

- **Always maximize.** Anything minimized (a loss, Branin) is negated once, at the objective boundary. All internal math assumes maximization.
- **Zero-mean GP prior**, made honest by standardizing y (see conditioning below).
- **Predicted std refers to the latent function f**, matching sklearn's convention when noise is passed via `alpha`. `include_noise=True` adds σ_n² to the predictive variance for noisy-observation bands.

### GP posterior (Cholesky only; K is never inverted)

Fit:

```text
K = k(X, X) + σ_n²·I + jitter·I
L = cholesky(K)                      # lower triangular
α = cho_solve((L, lower=True), y)
```

Predict at X*:

```text
μ*   = k(X*, X) · α
V    = solve_triangular(L, k(X, X*), lower=True)
var* = k(x*, x*) − diag(VᵀV)         # clipped at ≥ 1e-12 before sqrt
cov* = k(X*, X*) − VᵀV               # when return_cov=True (used by sample_posterior)
```

Jitter starts at 1e-10 and escalates ×10 on a failed factorization, up to 1e-6, then raises. Escalation is logged — never silent.

### Log marginal likelihood and hyperparameter fitting

```text
LML = −½·yᵀα − Σᵢ log Lᵢᵢ − (n/2)·log 2π
      [data fit]  [complexity]   [constant]
```

`fit_hyperparameters` maximizes LML over θ = log(ℓ, σ_f², σ_n²) using `scipy.optimize.minimize` (L-BFGS-B) with finite-difference gradients — adequate for three parameters at n ≤ ~40. Starts: the current θ (warm start) plus `n_restarts` draws log-uniform within bounds from the provided RNG; the best final LML wins. Default bounds, in normalized-input/standardized-y space: ℓ ∈ [0.01, 10], σ_f² ∈ [0.01, 100], σ_n² ∈ [1e-8, 1]. The analytic LML gradient is derived in the walkthrough document as a noted extension but is not implemented.

### Expected Improvement (closed form, maximization)

```text
I  = μ − y_best − ξ
z  = I / σ
EI = I·Φ(z) + σ·φ(z)    where σ > 1e-12; otherwise EI = 0
```

Result clipped at ≥ 0. Φ and φ from `scipy.stats.norm`. Default ξ = 0.01, interpreted in standardized-y units. y_best is the best *observed* standardized value (the standard noiseless-EI convention; the noisy-objective caveat is documented in the walkthrough).

### Data conditioning (optimizer's responsibility)

- Inputs mapped to [0,1]^d via the bounds; all GP and acquisition work happens there.
- y standardized to zero mean, unit variance before fitting (a zero-mean prior on raw accuracies ≈ 0.9 would pull predictions toward 0 in unexplored regions and distort EI).
- All reported results are transformed back to original units.

## 5. Bayesian optimization loop

1. Evaluate `n_init` seeded uniform-random points in the unit box (n_init = 3 for 1D demos, 5 for 2D).
2. Per iteration:
   1. Fit the GP on all observations so far.
   2. Re-fit kernel hyperparameters by maximizing LML (every iteration; warm start + restarts).
   3. Maximize EI over the unit box → x_next.
   4. Evaluate the true objective at x_next; append the observation; record history.
3. Repeat for `n_iter` iterations.

**EI maximization strategy:** in 1D, a dense 1000-point grid (the same grid the plots use). In 2–3D, 2048 seeded random candidates → evaluate EI → take the top 5 → refine each with L-BFGS-B on −EI within bounds → best result wins.

**Duplicate guard:** if the proposed point lies within 1e-6 of an existing sample (EI has collapsed to ~0 everywhere), fall back to a random point. Prevents stalling on repeats.

## 6. Experiments

### 6.1 `gp_demo.py` — the GP itself (outline Phase 4)

True function f(x) = x·sin(x) on [0, 10]. Three-panel figure: (a) samples from the GP prior; (b) posterior after 3 observations — mean, ±2σ band, posterior samples; (c) posterior after 8 observations. Prints the side-by-side sklearn comparison (fixed θ) and our fitted LML vs. sklearn's fitted LML. Also prints a **non-blocking hyperparameter-recovery sanity experiment**: fit on data sampled from a GP with known length scale and report the recovered value (informational output only; not an assertion — see §7). Output: `figures/gp_demo.png`.

### 6.2 `synthetic_optimization.py` — BO made visible (outline Phases 6–7)

- **1D:** maximize f(x) = −sin(3x) − x² + 0.7x on [−1, 2] (local + global maxima). Per-iteration two-panel frame — top: true f, GP mean ± 2σ, samples, next point; bottom: EI with its argmax marked. Output: `figures/bo_1d_iter_XX.png`, every frame.
- **2D:** Branin (negated to fit the maximize convention; classic benchmark with three global optima and known f*). Figures: true contours with the sample sequence; final GP mean contours; regret curve |best − f*| vs. iteration on a log scale. Output: `figures/bo_branin_*.png`.

### 6.3 `hyperparameter_tuning.py` — the real ML experiment (outline Phases 8–9)

- **Data:** `sklearn.datasets.load_digits` (1797 × 64), pixel values divided by 16. Stratified, seeded split: 80% CV pool / 20% held-out test.
- **Objective:** mean 5-fold stratified CV accuracy of `SVC(C=10^a, gamma=10^b)` on the pool. CV smooths the noise and plateaus of a single split.
- **Search box:** a = log₁₀C ∈ [−3, 3]; b = log₁₀γ ∈ [−5, 1]. BO operates in these log coordinates — the GP never sees raw C or γ.
- **Budget:** 25 evaluations per method per trial. BO: 5 initial + 20 iterations. Random search: 25 seeded uniform draws from the same box.
- **Trials:** 10 seeds per method. Estimated runtime ~10–25 minutes for the comparison itself, plus a one-time ~10-minute computation of the cached ground-truth grid (reused thereafter).
- **Figures:** best-so-far CV accuracy vs. evaluation count with mean ± std bands per method; BO's sampled points over the (a, b) plane on a cached 20×20 ground-truth accuracy grid (computed once, saved to `data/`).
- **Final honesty check:** each method's overall best configuration is retrained on the full pool and scored once on the held-out test set → reported in the README table, verifying that CV gains transfer.
- **Honest framing:** BO is expected to win clearly in early evaluations; random search often nearly catches up by evaluation 25 on a well-bounded 2D space (Bergstra & Bengio). The README frames the result around evaluation efficiency, as the outline already does.

## 7. Testing and validation

- **`test_kernels.py`:** symmetry K = Kᵀ; k(x, x) = σ_f²; positive semidefiniteness (min eigenvalue ≥ −1e-8 on random input sets); values decay monotonically with distance; correct shapes for 1D and 2D inputs.
- **`test_gp.py`:**
  - vs. sklearn — the benchmark test: identical kernel parameters, noise via `alpha`, `optimizer=None`, `normalize_y=False` → predicted mean and std `allclose` at atol 1e-6.
  - our LML value vs. sklearn's `log_marginal_likelihood()` at identical θ.
  - a near-noiseless GP interpolates its training data.
  - far from data, the posterior reverts to the prior (mean → 0, std → σ_f).
  - **LML optimization test (deterministic):** on fixed seeded data, after `fit_hyperparameters`, LML(θ_opt) ≥ LML(θ_init), and all fitted parameters are finite and within bounds. Hyperparameter *recovery* is deliberately not asserted (MLE variance makes it flaky); it lives as the non-blocking sanity experiment in `gp_demo.py` (§6.1).
- **`test_acquisition.py`:** EI ≥ 0 everywhere; EI = 0 when σ = 0; EI grows with σ at fixed μ; closed form matches a fixed-seed Monte-Carlo estimate using ~1e5 draws (deterministic given the seed; rtol ~1e-2).
- **`test_optimizer.py`:** seeded BO finds the maximum of a smooth 1D function within tolerance in ≤ 15 evaluations; same seed → identical result; the duplicate guard triggers when EI collapses.

## 8. Documentation

- **`docs/math-walkthrough.md`** — written alongside the code, one section per module: multivariate-Gaussian conditioning → the posterior equations; the LML derivation and its Occam's-razor reading; the EI closed-form derivation; why Cholesky instead of inversion; numerics notes (jitter, standardization, log-space search); the analytic LML gradient as a derived-but-unimplemented extension; the noisy-objective EI caveat.
- **`README.md`** — the project story with embedded figures, the results table (including the held-out test check), how to run everything with uv, limitations (O(n³) scaling, low-dimensional focus), and references (Rasmussen & Williams, *Gaussian Processes for Machine Learning*; Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*).
- Code is annotated at derivation grade: each core routine's comments state the equation being computed and why the numerically stable form is used.

## 9. Risks and known caveats

- **LML is multimodal** in θ; multi-start L-BFGS mitigates but does not guarantee the global optimum. The deterministic test asserts improvement, not global optimality.
- **EI with a noisy objective** uses best-observed-y, which is slightly optimistic; 5-fold CV keeps the noise small at this scale. Documented, not "solved" — noise-robust EI variants are out of scope.
- **Random search is a strong baseline** on a well-bounded 2D space; the comparison is framed around evaluation efficiency, and results are reported honestly across 10 seeds either way.
