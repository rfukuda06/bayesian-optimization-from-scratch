# Reusable ML hyperparameter-tuning adapter — design

**Date:** 2026-08-09
**Status:** approved in brainstorm (three design sections signed off individually)
**Baseline:** extends the completed GP+BO project at tip `191405e`
**Companion docs:** `docs/superpowers/specs/2026-08-09-gp-bayesian-optimization-design.md` (original spec), `docs/math-walkthrough.md` (unchanged by this extension)

## Decisions log

- **Option B chosen** (over leaving the digits experiment untouched): the flagship
  digits experiment is refactored to consume the new adapter, so the benchmark
  itself demonstrates reusability. Fidelity is proven by an exact-equivalence unit
  test plus one full re-run whose bit-deterministic figures must `git diff` clean.
- `n_init`/`n_iter` mirror `BayesianOptimizer.run` directly — no single `n_evals`
  knob, no invented split policy.
- `cv` accepts an int (→ seeded, shuffled `StratifiedKFold`) **or** a pre-built
  sklearn splitter used as-is. No classification/regression detection anywhere;
  the splitter form is the regression escape hatch.
- `scoring=None` default → estimator's own scorer (exactly what the current
  experiment does). Any sklearn scoring string/scorer is passed through.
- Maximization only, matching the optimizer; minimize via sklearn `neg_*` scorers.
- Demo: `load_breast_cancer` + `Pipeline(StandardScaler, LogisticRegression)`,
  tuning 1D `log10_C`.
- Documentation lands in the README only; the math walkthrough is untouched
  (this module contains no new math).

## Purpose and success criteria

The core BO library already optimizes any `objective(x) -> score` over box bounds.
This extension adds a thin sklearn adapter so a dataset + model factory +
parameter space + scoring metric become such an objective without rewriting the
cross-validation plumbing per project. It converts the repo from "library plus
experiments" into "library with a demonstrated reusable API." It is explicitly
**not** AutoML.

Success means all of the following hold:

1. `uv run pytest` passes: all 26 existing tests unchanged and green, plus ~7 new
   tests in `tests/test_model_selection.py`.
2. After the digits-experiment refactor, a full re-run reproduces
   `figures/hp_comparison.png` and `figures/hp_landscape.png` byte-identically
   (clean `git diff`) and prints checkpoint numbers matching the README table
   (see verification protocol below).
3. `experiments/generic_tuning_demo.py` runs end-to-end in under about a minute,
   prints tuned vs. untuned-baseline CV scores, and writes
   `figures/generic_tuning_demo.png`.
4. `from gpbo import tune_model, TuningResult, build_cv_objective, decode_parameters`
   works; `grep` confirms `kernels.py`, `gp.py`, `acquisition.py`, `optimizer.py`
   still import no sklearn.
5. README gains a "Reusable model tuning" section; `pyproject.toml` and
   `docs/math-walkthrough.md` are unchanged.

## Scope

**In scope:** one new module `src/gpbo/model_selection.py`; a tiny refactor of
`experiments/hyperparameter_tuning.py` onto it; one lightweight demo
`experiments/generic_tuning_demo.py`; `tests/test_model_selection.py`; a README
section; package exports.

**Out of scope (deliberately):** CSV/preprocessing automation, target-column
detection, missing-value handling, categorical encoding, automatic feature
scaling, automatic model selection, classification/regression detection,
neural-network training, categorical/conditional/integer hyperparameters,
distributed or parallel BO, GUIs, AutoML. The caller prepares `X, y` and defines
the factory and search space. Also unchanged: the four core modules, the plan's
deferred math list (Matérn, UCB, GIFs, analytic LML gradients) remains deferred.

## Architecture

```
core (pure NumPy/SciPy, sklearn-free):  kernels.py  gp.py  acquisition.py  optimizer.py
                                            ↑
ML adapter (imports sklearn + optimizer):  model_selection.py        # NEW
                                            ↑
experiments:  hyperparameter_tuning.py (refactored consumer)
              generic_tuning_demo.py (new consumer)
```

Import direction is one-way: `model_selection` imports from `gpbo.optimizer`;
core modules never import `model_selection` or sklearn. scikit-learn is already a
core dependency in `pyproject.toml` (used by experiments and agreement tests), so
packaging does not change — the boundary being preserved is a code-import
boundary, not a dependency change.

## API specification — `src/gpbo/model_selection.py`

```python
@dataclass
class TuningResult:
    best_params: dict            # e.g. {"log10_C": 0.38, "log10_gamma": -0.91}
    best_cv_score: float         # = optimization_result.best_y
    optimization_result: OptimizationResult   # full BO history

def decode_parameters(x, param_names) -> dict

def build_cv_objective(X, y, model_factory, param_names, cv,
                       scoring=None, n_jobs=None) -> Callable

def tune_model(X, y, model_factory, param_space, scoring=None,
               cv=5, n_init=5, n_iter=20, seed=0, n_jobs=None) -> TuningResult
```

Semantics:

- **`decode_parameters(x, param_names)`** zips names onto the optimizer's vector
  in position order and coerces values to Python `float`. The insertion order of
  `param_space` defines dimension order (guaranteed for dicts since Python 3.7;
  documented and tested).
- **`build_cv_objective`** returns `objective(x) -> float`: decode `x`, call
  `model_factory(params)`, return `cross_val_score(model, X, y, cv=cv,
  scoring=scoring, n_jobs=n_jobs).mean()`. `cv` is used exactly as given — this
  lower-level builder applies no seeding convenience. The returned objective
  accepts any 1-D array-like (the optimizer passes `(d,)` arrays in original
  units; the experiment's random-search arm passes ndarray rows).
- **`tune_model`** validates the space, resolves `cv`, builds the objective,
  runs `BayesianOptimizer(objective, bounds).run(n_init, n_iter, seed=seed)`,
  and wraps the result:
  - bounds: `np.array(list(param_space.values()), dtype=float)` → shape `(d, 2)`.
  - `cv` int → `StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)`;
    splitter instance → used as-is (then `seed` drives only the BO run).
  - `best_params = decode_parameters(result.best_x, names)`;
    `best_cv_score = result.best_y`.
- **Validation (kept minimal):** `ValueError` for an empty `param_space` and for
  any bounds pair with `lo >= hi`; everything else defers to sklearn's own
  errors.
- **Transforms are caller-side:** the search space is whatever the factory
  expects (e.g. `log10_C`), and the factory applies `10**x`. This keeps the GP
  modeling well-scaled variables and keeps the BO library mathematically simple.
- **Exports:** all four public names added to `gpbo/__init__.py` and `__all__`.
- **Annotation register:** design rationale in docstrings/comments — why a fixed
  splitter makes `f(x)` deterministic (otherwise re-evaluating the same
  hyperparameters returns different scores, i.e. artificial observation noise),
  why transforms belong to the caller, why maximization-only — matching the
  project's teaching style. No derivations; no math-walkthrough changes.

## Digits experiment refactor — `experiments/hyperparameter_tuning.py`

Only the objective construction changes.

Before (current code):

```python
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def objective(params):
    a, b = params
    clf = SVC(C=10.0**a, gamma=10.0**b)
    return cross_val_score(clf, X_pool, y_pool, cv=CV, n_jobs=-1).mean()
```

After:

```python
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def make_svc(params):
    return SVC(C=10.0**params["log10_C"], gamma=10.0**params["log10_gamma"])

objective = build_cv_objective(
    X_pool, y_pool, model_factory=make_svc,
    param_names=("log10_C", "log10_gamma"), cv=CV, n_jobs=-1,
)
```

Untouched: `BOUNDS`, `N_SEEDS = 10`, `N_INIT/N_ITER = 5/20`, the BO arm, the
random-search arm (it calls the same `objective`), the landscape cache
(`data/digits_landscape.npz`), all plotting, and the held-out test. The
fixed-folds-across-all-seeds behavior is preserved because the experiment passes
its own `random_state=0` splitter; per-seed randomness continues to drive only
the optimizer and random-search draws.

Floats are identical by construction: same `SVC` construction, same splitter
instance, same `cross_val_score(...).mean()` call — the only code difference is
dict lookup versus tuple unpacking of the two parameters.

### Verification protocol (in order)

1. **Unit test first:** `build_cv_objective` output asserted **exactly equal**
   (`==`, not `allclose`) to a hand-rolled `cross_val_score` objective at
   several points on a small dataset.
2. **Full re-run** of the experiment after the refactor (~10–25 min; the
   landscape cache already exists, so no grid recomputation).
3. **Proof:** clean `git diff` on `figures/hp_comparison.png` and
   `figures/hp_landscape.png` (the figure pipeline is bit-deterministic), and
   printed checkpoints matching the README ground truth — BO/RS mean best CV
   accuracy 0.9837/0.9820 after 5 evals, 0.9867/0.9872 after 10, 0.9887/0.9877
   after 25; held-out test 0.9889 for both.
4. **If the diff is not clean:** stop, diagnose systematically, and do not
   commit the refactor until the discrepancy is fully explained. No README
   number changes are expected or acceptable from this extension.

The refactor and its verification evidence land in one commit whose message
records the clean-diff proof.

## Generalization demo — `experiments/generic_tuning_demo.py`

Proves the same adapter works on a different dataset **and** estimator type,
without touching the optimizer:

- `load_breast_cancer()`; factory returns
  `Pipeline(StandardScaler(), LogisticRegression(C=10**params["log10_C"], max_iter=1000))`.
  A Pipeline *is* an estimator, so this demonstrates estimator-agnosticism for
  free and avoids convergence warnings from unscaled features.
- `param_space = {"log10_C": (-4.0, 4.0)}` — 1D, exercising the dense-grid EI
  path.
- Calls `tune_model` with its defaults (`cv=5, n_init=5, n_iter=20, seed=0`), so
  the demo doubles as documentation of default usage.
- Prints best params and best CV score plus the untuned baseline (`C=1`, same
  folds) for a one-line "did tuning help" comparison; saves a best-so-far curve
  to `figures/generic_tuning_demo.png`.
- Conventions match existing experiments: plain `if __name__ == "__main__":`,
  writes to `figures/`, module docstring stating what it proves. Runtime well
  under a minute.

## Testing — `tests/test_model_selection.py`

About 7 pytest tests, seconds total, on tiny data (`make_classification` with a
fixed `random_state`):

1. `decode_parameters` maps vector → names in order.
2. `build_cv_objective` output **exactly equals** a hand-rolled
   `cross_val_score` objective (the fidelity proof backing the refactor).
3. Determinism: the objective returns identical floats for the same `x` twice.
4. `tune_model` best_params and every evaluated point
   (`optimization_result.X`) lie within the requested bounds.
5. Same seed → fully reproducible run (`np.testing.assert_array_equal` on `X`
   and `y`, mirroring `test_bo_is_reproducible_for_same_seed`).
6. End-to-end smoke: tune a small logistic regression;
   `best_cv_score == max(optimization_result.y)` and `best_params` decodes
   `best_x`.
7. Validation: empty `param_space` raises `ValueError`; `lo >= hi` bounds raise
   `ValueError`.

No tests of sklearn itself. Existing 26 tests are not modified.

## Documentation

- **README:** new "Reusable model tuning" section placed after the
  hyperparameter-tuning results — 2–3 sentences of framing (objective-agnostic
  core, thin sklearn adapter), the short `tune_model` usage example
  (breast-cancer version), a note that the digits benchmark itself runs through
  `build_cv_objective`, and a pointer to the demo script. Add
  `uv run python experiments/generic_tuning_demo.py` to "How to run". The
  "What is implemented from scratch" section stays untouched — the adapter is
  plumbing, not from-scratch math.
- **Unchanged:** `docs/math-walkthrough.md`, `pyproject.toml` (no new
  dependencies).
- The implementation plan follows separately as `docs/ml-tuning-adapter-plan.md`
  (repo convention for plan files).

## Risks and known caveats

- **Bit-determinism assumption.** The verification hinges on the figure pipeline
  being bit-deterministic in the current environment (it was during the original
  project, including with `n_jobs=-1` — joblib returns fold scores in fold
  order, so parallel CV does not perturb the mean). If the re-run diff is dirty,
  the protocol says stop and diagnose; the refactor does not land unexplained.
- **Continuous-only spaces.** All dimensions are continuous floats by design;
  integer/categorical/conditional parameters are out of scope, and the README
  example should not suggest otherwise.
- **`cv=int` ties folds to `seed`.** Convenient for casual use, but anyone
  comparing tuning runs across seeds on a fixed landscape must pass an explicit
  splitter (exactly as the digits experiment does). The `tune_model` docstring
  must state this explicitly.
- **Touching a finished project.** The digits experiment is a completed,
  results-bearing artifact; the refactor is intentionally minimal and gated on
  the verification protocol above.
