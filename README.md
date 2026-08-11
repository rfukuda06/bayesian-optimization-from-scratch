# Gaussian Processes + Bayesian Optimization from scratch

Gaussian process regression and Bayesian optimization implemented from the math up in NumPy and SciPy — RBF kernel, Cholesky-based posterior, log-marginal-likelihood hyperparameter fitting, and closed-form Expected Improvement. The stack is exposed as a small reusable interface, `tune_model`, that tunes the hyperparameters of any scikit-learn estimator on any dataset you provide. The GP is validated against scikit-learn's `GaussianProcessRegressor` to `atol 1e-6` (agreement measured at ~`1e-9`), the acquisition function against a Monte-Carlo estimate, and the tuning loop is benchmarked against random search on the digits dataset. This is a learning project: the code is annotated at derivation grade and comes with a [full math walkthrough](docs/math-walkthrough.md).

## The pipeline

```mermaid
flowchart LR
    A[Gaussians] --> B[RBF kernel]
    B --> C[GP prior]
    C -->|condition on data| D[GP posterior]
    D -->|LML + L-BFGS-B| E[fit hyperparameters]
    E --> F[Expected Improvement]
    F --> G[BO loop]
    G --> H[SVM tuning]
    H --> I[vs random search]
```

A joint Gaussian over function values, with covariance supplied by the RBF kernel, is the prior. Conditioning on observations gives the posterior mean and variance. Fitting the kernel hyperparameters means maximizing the log marginal likelihood. Expected Improvement turns the posterior into a score that balances exploiting the mean against exploring the variance; the BO loop repeatedly maximizes it, evaluates the objective there, and refits. The final experiment points that loop at a real hyperparameter search and compares it to random search.

## Tune your model on your data

The optimizer is not tied to any dataset or model. You bring three things — your data as numeric arrays, a factory that builds your estimator, and bounds for the knobs you want tuned — and `tune_model` does the rest: a small adapter (`src/gpbo/model_selection.py`) turns fixed-fold cross-validation into a black-box objective, and the GP/Expected-Improvement loop spends 25 evaluations (by default) finding the best settings.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpbo import tune_model

df = pd.read_csv("my_data.csv")                    # any dataset you have prepared
X = df.drop(columns="outcome").values
y = df["outcome"].values

def make_model(params):                            # you choose the estimator...
    return Pipeline([("scale", StandardScaler()),
                     ("logreg", LogisticRegression(C=10.0 ** params["log10_C"], max_iter=1000))])

result = tune_model(X, y, model_factory=make_model,
                    param_space={"log10_C": (-4.0, 4.0)},   # ...and which knobs to search
                    cv=5, seed=0)

result.best_params                       # {"log10_C": -0.35} — plug back into your factory
result.best_cv_score                     # best mean CV accuracy found
result.optimization_result.best_so_far   # convergence curve, if you want to plot it
```

Fitting the final model is yours: call your factory with `result.best_params` and train on your full training split — the library tunes, you deploy.

The division of labor is deliberate:

- **You choose the model.** Any scikit-learn estimator works (a `Pipeline` counts); the library never auto-selects a model type. Comparing, say, an SVM against a random forest means two `tune_model` calls and two scores — this is a tuning library, not AutoML.
- **You own the transforms.** Searching `log10_C` and applying `10**x` inside the factory keeps the GP modeling a well-scaled space.
- **You prepare the data.** `X, y` must already be numeric arrays; there is no CSV cleaning, encoding, or missing-value handling here.
- **Continuous knobs only.** Bounds are float ranges; categorical or conditional hyperparameters are out of scope.
- The CV folds are fixed for the whole run, so the objective is deterministic; scores are maximized (use sklearn's `neg_*` scorers to minimize a loss).

`experiments/generic_tuning_demo.py` is exactly this recipe run end to end on scikit-learn's `breast_cancer` dataset (569 tumor samples, 30 features) — to tune your own data, copy that file and swap the loading lines and the factory. It finishes in seconds: on the committed run (seed 0), tuning lifts a scaled logistic regression from the untuned `C = 1` baseline's `0.9789` mean 5-fold CV accuracy to `0.9824` at `C = 10^-0.35`:

![Reusable tuning demo](figures/generic_tuning_demo.png)

## The benchmark: BO vs random search on digits

The demo above shows the interface; this experiment is the evidence that the optimizer underneath earns its keep. A single tuning run takes under a minute — this script runs **twenty** of them (10 seeds × {Bayesian optimization, random search} = 500 evaluations, all through the same `build_cv_objective` adapter), which is why it takes ~15 minutes: the point is a fair, seed-averaged comparison with error bars, not one lucky run.

Tuning `SVC(C, γ)` on scikit-learn's digits dataset. Search space is `log₁₀C ∈ [-3, 3]`, `log₁₀γ ∈ [-5, 1]`; the objective is mean 5-fold stratified CV accuracy on an 80% pool. Each method gets 25 evaluations per seed, averaged over 10 seeds. A held-out 20% test set is touched exactly once per method at the end.

![BO vs random search](figures/hp_comparison.png)

| checkpoint | BO (mean best CV acc) | Random search (mean best CV acc) |
|---|---|---|
| after 5 evals | 0.9837 | 0.9820 |
| after 10 evals | 0.9867 | 0.9872 |
| after 25 evals | 0.9887 | 0.9877 |

Held-out test accuracy of each method's overall-best config:

```
BO best config: C=10^0.38, gamma=10^-0.91  -> held-out test accuracy 0.9889
RS best config: C=10^0.34, gamma=10^-0.60  -> held-out test accuracy 0.9889
```

**Reading these honestly.** BO leads at 5 evaluations, random search is marginally ahead at 10 (`0.9872` vs `0.9867`), and BO is back in front by 25. The first 5 evaluations of *both* methods are random points — BO has not consulted its GP yet — so the 5-eval gap is partly seed luck, and the numbers stay within a fraction of a percent of each other throughout. This is the expected picture on an easy, well-bounded 2D landscape with a broad high-accuracy plateau: random search is genuinely competitive here, and both methods find configs that tie at `0.9889` on held-out test (Bergstra & Bengio, 2012, make exactly this point about random search on low-effective-dimension spaces). BO's advantage is sample efficiency early, and it grows on more expensive or more structured objectives where every evaluation is costly. Where BO concentrates its samples on the CV-accuracy landscape:

![Where BO samples](figures/hp_landscape.png)

## Under the hood: GP regression and synthetic BO

Before any real tuning, the building blocks are exercised on problems where the truth is known: GP regression on a toy function, then BO on synthetic objectives.

### GP regression

Prior samples, then the posterior after 3 and 8 noisy observations of `f(x) = x sin(x)`, with hyperparameters fitted by maximizing the LML. The `±2σ` band collapses near data and widens back to the prior away from it.

![GP prior and posterior](figures/gp_demo.png)

### 1D Bayesian optimization

Two frames from a 12-iteration run maximizing `-sin(3x) - x² + 0.7x`. Top panel: true objective, GP mean, `±2σ`, samples so far, and the next point (red). Bottom panel: the EI surface whose argmax picks that next point. Early on EI probes the uncertain regions; by the late frame the posterior has locked onto the optimum and EI has flattened.

| Iteration 1 (early) | Iteration 12 (late) |
|---|---|
| ![BO iteration 1](figures/bo_1d_iter_01.png) | ![BO iteration 12](figures/bo_1d_iter_12.png) |

### 2D Bayesian optimization (Branin)

Sample placement over the Branin function (three global minima). The optimizer concentrates its evaluations in one minimum's basin. On the committed seed-0 run it reached a best value of `1.038` against the true global minimum of `0.398`; other seeds landed in the `0.46`–`0.72` range. The figure shows honest basin concentration, not an exact hit on the global minimum.

![Branin sample sequence](figures/bo_branin_samples.png)

## What is implemented from scratch

- **RBF kernel** — `k(x, x') = σ_f² exp(-‖x - x'‖² / 2ℓ²)`, with the squared-distance expansion done without an `(n, m, d)` intermediate (`src/gpbo/kernels.py`).
- **GP posterior via Cholesky** — `α = K⁻¹y` through `cho_solve` (never an explicit inverse), predictive variance from triangular solves, jitter escalation for stability (`src/gpbo/gp.py`).
- **Log marginal likelihood + multi-start fitting** — the LML in closed form using the Cholesky factor for `log|K|`, maximized over `(ℓ, σ_f², σ_n²)` in log space with multi-start L-BFGS-B; the current point is always kept as a candidate so the fit never worsens the starting LML (`src/gpbo/gp.py`).
- **Expected Improvement** — the closed form `EI = I·Φ(z) + σ·φ(z)`, with the `σ→0` point-mass case handled explicitly (`src/gpbo/acquisition.py`).
- **BO loop** — standardization of `y`, input scaling to `[0,1]^d`, warm-started hyperparameters across iterations, grid EI maximization in 1D and candidate + L-BFGS-B refinement in higher dimensions, plus a duplicate guard so the loop cannot stall re-evaluating a point (`src/gpbo/optimizer.py`).

Delegated to the numerical libraries: dense linear algebra primitives (`scipy.linalg.cholesky`, `cho_solve`, `solve_triangular`), the L-BFGS-B optimizer (`scipy.optimize.minimize`), and the standard-normal `pdf`/`cdf` (`scipy.stats.norm`). The experiments use scikit-learn for the digits data and the SVM, and for the reference `GaussianProcessRegressor` in the agreement tests.

## Correctness

- **GP agreement with scikit-learn** — at fixed hyperparameters, our posterior mean, posterior std, and log marginal likelihood match `GaussianProcessRegressor` to `atol 1e-6`. Measured agreement is around `1e-9`; the demo script prints the live `max|Δmean|` and `max|Δstd|`.
- **EI against Monte Carlo** — the closed-form EI is checked against the mean of `100,000` draws from `N(μ, σ²)` passed through `max(f - y_best - ξ, 0)`, agreeing to `rtol 2e-2`.

```
$ uv run pytest
34 passed
```

## How to run

```bash
uv sync                                             # create the env from pyproject / uv.lock
uv run pytest                                       # 34 passed

uv run python experiments/gp_demo.py                # GP figures + agreement prints (seconds)
uv run python experiments/synthetic_optimization.py # 1D frames, Branin figures (~1–2 min)
uv run python experiments/hyperparameter_tuning.py  # BO vs RS on digits (~15–35 min first run)
uv run python experiments/generic_tuning_demo.py    # reusable-tuning demo (<1 min)
```

The hyperparameter experiment is slow on its first run because it computes a `20×20` ground-truth CV-accuracy grid for the landscape plot. That grid is cached to `data/digits_landscape.npz` (committed), so subsequent runs skip it and finish much faster.

## Limitations

- **`O(n³)` scaling.** The Cholesky factorization is cubic in the number of observations, which is fine for the tens-to-hundreds of points a BO run accumulates but rules out large-`n` regression without sparse or inducing-point approximations.
- **Low-dimensional scope.** Everything here is exercised in 1–3 dimensions. EI over a box gets progressively harder to maximize as dimension grows, and this repo does not implement the trust-region or high-dimensional acquisition machinery that addresses it.
- **Noisy-EI caveat.** EI uses the best *observed* `y` as its incumbent. Under observation noise the true incumbent is uncertain, and plain EI can be over-optimistic; a noise-aware acquisition (e.g. expected improvement over the posterior mean, or knowledge gradient) would be the principled fix.

## References

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006) — ch. 2 (regression, Cholesky prediction) and ch. 5 (model selection, marginal likelihood).
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13 (2012) — why random search is a strong baseline on spaces with low effective dimension.
- [Design spec](docs/superpowers/specs/2026-08-09-gp-bayesian-optimization-design.md) — decisions, scope, and success criteria.
- [Math walkthrough](docs/math-walkthrough.md) — every equation this library implements, derived end to end, each section pointing at the code that implements it.
