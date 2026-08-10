# Reusable ML Tuning Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin sklearn adapter (`tune_model` / `build_cv_objective` / `decode_parameters` / `TuningResult`) over the unchanged BO core, refactor the digits benchmark onto it with an exact-equivalence verification, and add a breast-cancer generalization demo.

**Architecture:** One new module `src/gpbo/model_selection.py` bridges sklearn and `BayesianOptimizer`; import direction is one-way (adapter → optimizer) and the four core modules stay sklearn-free. The digits experiment swaps only its objective construction onto `build_cv_objective`; a new demo proves the interface on a different dataset and estimator type. Spec: `docs/superpowers/specs/2026-08-09-ml-tuning-adapter-design.md`.

**Tech Stack:** Python ≥3.11, NumPy, scikit-learn ≥1.4 (already a dependency — `pyproject.toml` does not change), pytest, matplotlib (Agg), `uv` for env/commands.

**Conventions that apply to every task:**
- Run everything through `uv run ...` from the repo root.
- Commits go directly to `main` (repo convention), conventional-commit style, each ending with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Comment register: docstrings carry design rationale (why fixed folds, why caller-side transforms); inline comments are short and factual, matching `src/gpbo/gp.py`.
- Baseline before Task 1: `uv run pytest` → `26 passed`.

---

### Task 1: `decode_parameters` + module skeleton

**Files:**
- Create: `tests/test_model_selection.py`
- Create: `src/gpbo/model_selection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_selection.py`:

```python
import numpy as np

from gpbo.model_selection import decode_parameters


def test_decode_parameters_maps_names_in_order():
    params = decode_parameters(np.array([0.5, -1.5]), ("log10_C", "log10_gamma"))
    assert params == {"log10_C": 0.5, "log10_gamma": -1.5}
    assert all(type(v) is float for v in params.values())
    assert list(params) == ["log10_C", "log10_gamma"]   # dimension order preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: error during collection — `ModuleNotFoundError: No module named 'gpbo.model_selection'`

- [ ] **Step 3: Write minimal implementation**

Create `src/gpbo/model_selection.py`:

```python
"""Bridge between scikit-learn model selection and the generic Bayesian optimizer.

The core library (`gp`, `acquisition`, `optimizer`) knows nothing about sklearn:
it maximizes an arbitrary objective(x) over a box. This module turns

    (X, y) + model factory + parameter space + CV scheme

into exactly such an objective, so tuning any sklearn estimator is one call.

Design notes, in the spirit of the rest of the package:

- The CV splitter is FIXED for a whole run. Re-evaluating the same
  hyperparameters must return the same score; with re-drawn folds the
  objective would carry artificial observation noise on top of the true
  landscape, which the GP would then have to absorb into its fitted noise
  variance. A fixed splitter makes f(x) deterministic.
- Transforms (e.g. C = 10**log10_C) belong to the caller's model factory.
  The optimizer then searches a well-scaled space (log10_C in [-3, 3])
  rather than a wildly skewed one (C in [1e-3, 1e3]) where a single RBF
  length-scale per dimension would fit poorly.
- Scores are MAXIMIZED, matching BayesianOptimizer. To minimize a loss,
  use one of sklearn's negated scorers (e.g. scoring="neg_mean_squared_error").
"""

import numpy as np


def decode_parameters(x, param_names) -> dict:
    """Map the optimizer's vector x (d,) onto named parameters, in order.

    Dimension i of the search space is param_names[i]; for `tune_model` that
    order is the insertion order of `param_space` (guaranteed for dicts since
    Python 3.7). Values are coerced to plain floats so results print and
    serialize cleanly.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) != len(param_names):
        raise ValueError(f"got {len(x)} values for {len(param_names)} parameters")
    return {name: float(v) for name, v in zip(param_names, x)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/model_selection.py tests/test_model_selection.py
git commit -m "feat: model_selection module with decode_parameters

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `build_cv_objective`

**Files:**
- Modify: `tests/test_model_selection.py`
- Modify: `src/gpbo/model_selection.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_model_selection.py`, replace the import block with:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gpbo.model_selection import build_cv_objective, decode_parameters

X_SMALL, Y_SMALL = make_classification(
    n_samples=80, n_features=6, n_informative=4, random_state=0
)


def _logreg_factory(params):
    return LogisticRegression(C=10.0 ** params["log10_C"], max_iter=200)
```

and append at the end of the file:

```python
def test_build_cv_objective_matches_hand_rolled_cross_val_score():
    # The fidelity proof behind the digits-experiment refactor: the adapter
    # must produce the exact floats the hand-rolled objective produced.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    objective = build_cv_objective(
        X_SMALL, Y_SMALL, model_factory=_logreg_factory,
        param_names=("log10_C",), cv=cv,
    )
    for c in (-1.0, 0.0, 1.5):
        clf = LogisticRegression(C=10.0**c, max_iter=200)
        expected = cross_val_score(clf, X_SMALL, Y_SMALL, cv=cv).mean()
        assert objective(np.array([c])) == expected   # exact, not allclose


def test_objective_is_deterministic_for_same_x():
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    objective = build_cv_objective(
        X_SMALL, Y_SMALL, model_factory=_logreg_factory,
        param_names=("log10_C",), cv=cv,
    )
    x = np.array([0.3])
    assert objective(x) == objective(x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: collection error — `ImportError: cannot import name 'build_cv_objective'`

- [ ] **Step 3: Write minimal implementation**

In `src/gpbo/model_selection.py`, change the import block to:

```python
import numpy as np
from sklearn.model_selection import cross_val_score
```

and append after `decode_parameters`:

```python
def build_cv_objective(X, y, model_factory, param_names, cv,
                       scoring=None, n_jobs=None):
    """Return objective(x) -> mean CV score, ready for BayesianOptimizer.

    `cv` is used exactly as given (anything `cross_val_score` accepts). Pass a
    splitter with a fixed random_state to make the objective deterministic —
    the module docstring explains why that matters for the GP. `scoring=None`
    delegates to the estimator's default scorer; any sklearn scoring string or
    scorer is passed straight through.
    """
    def objective(x):
        params = decode_parameters(x, param_names)
        model = model_factory(params)
        return cross_val_score(
            model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
        ).mean()

    return objective
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/model_selection.py tests/test_model_selection.py
git commit -m "feat: build_cv_objective turns sklearn CV into a BO objective

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `TuningResult` + `tune_model`

**Files:**
- Modify: `tests/test_model_selection.py`
- Modify: `src/gpbo/model_selection.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_model_selection.py`, add `import pytest` after `import numpy as np`, extend the `gpbo.model_selection` import to:

```python
from gpbo.model_selection import (
    TuningResult,
    build_cv_objective,
    decode_parameters,
    tune_model,
)
```

and append at the end of the file:

```python
def test_tune_model_stays_within_bounds():
    result = tune_model(
        X_SMALL, Y_SMALL, model_factory=_logreg_factory,
        param_space={"log10_C": (-2.0, 2.0)}, cv=3, n_init=3, n_iter=3, seed=0,
    )
    assert -2.0 <= result.best_params["log10_C"] <= 2.0
    X_evals = result.optimization_result.X
    assert np.all((X_evals >= -2.0) & (X_evals <= 2.0))


def test_tune_model_same_seed_reproducible():
    kwargs = dict(
        model_factory=_logreg_factory, param_space={"log10_C": (-2.0, 2.0)},
        cv=3, n_init=3, n_iter=3, seed=1,
    )
    r1 = tune_model(X_SMALL, Y_SMALL, **kwargs)
    r2 = tune_model(X_SMALL, Y_SMALL, **kwargs)
    np.testing.assert_array_equal(
        r1.optimization_result.X, r2.optimization_result.X
    )
    np.testing.assert_array_equal(
        r1.optimization_result.y, r2.optimization_result.y
    )


def test_tune_model_end_to_end_smoke():
    result = tune_model(
        X_SMALL, Y_SMALL, model_factory=_logreg_factory,
        param_space={"log10_C": (-2.0, 2.0)}, cv=3, n_init=3, n_iter=3, seed=0,
    )
    assert isinstance(result, TuningResult)
    opt = result.optimization_result
    assert result.best_cv_score == opt.y.max()
    assert result.best_params == decode_parameters(opt.best_x, ("log10_C",))
    assert opt.X.shape == (6, 1)   # n_init + n_iter evaluations


def test_validation_errors():
    with pytest.raises(ValueError):
        tune_model(X_SMALL, Y_SMALL, model_factory=_logreg_factory,
                   param_space={})
    with pytest.raises(ValueError):
        tune_model(X_SMALL, Y_SMALL, model_factory=_logreg_factory,
                   param_space={"log10_C": (2.0, -2.0)})
    with pytest.raises(ValueError):
        decode_parameters(np.array([1.0, 2.0]), ("only_one",))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: collection error — `ImportError: cannot import name 'TuningResult'`

- [ ] **Step 3: Write minimal implementation**

In `src/gpbo/model_selection.py`, change the import block to:

```python
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gpbo.optimizer import BayesianOptimizer, OptimizationResult
```

and append at the end of the file:

```python
@dataclass
class TuningResult:
    """Best hyperparameters by name, their CV score, and the full BO history."""

    best_params: dict
    best_cv_score: float
    optimization_result: OptimizationResult


def tune_model(X, y, model_factory, param_space, scoring=None, cv=5,
               n_init=5, n_iter=20, seed=0, n_jobs=None) -> TuningResult:
    """Tune `model_factory`'s hyperparameters over `param_space` with BO.

    `param_space` maps names to continuous (lo, hi) bounds, e.g.
    {"log10_C": (-3.0, 3.0)}; its insertion order defines the optimizer's
    dimension order. The factory receives {name: float} and returns a fresh
    unfitted estimator (a Pipeline counts) — transforms like C = 10**log10_C
    live there, not here.

    `cv` as an int becomes StratifiedKFold(cv, shuffle=True, random_state=seed)
    — a classification default whose folds are tied to `seed`. To hold folds
    fixed while varying `seed` (as experiments/hyperparameter_tuning.py does
    across its trials), pass an explicit splitter instead; regression callers
    pass e.g. KFold. `n_init`/`n_iter` mirror BayesianOptimizer.run: n_init
    random evaluations, then n_iter EI-guided ones.
    """
    if not param_space:
        raise ValueError("param_space must contain at least one parameter")
    for name, (lo, hi) in param_space.items():
        if not lo < hi:
            raise ValueError(
                f"bounds for {name!r} must satisfy lo < hi, got ({lo}, {hi})"
            )
    names = tuple(param_space)
    bounds = np.array([param_space[n] for n in names], dtype=float)
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    objective = build_cv_objective(
        X, y, model_factory, names, cv, scoring=scoring, n_jobs=n_jobs
    )
    result = BayesianOptimizer(objective, bounds).run(n_init, n_iter, seed=seed)
    return TuningResult(
        best_params=decode_parameters(result.best_x, names),
        best_cv_score=result.best_y,
        optimization_result=result,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: `7 passed`

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: `33 passed` (26 existing + 7 new; no existing test modified)

- [ ] **Step 6: Commit**

```bash
git add src/gpbo/model_selection.py tests/test_model_selection.py
git commit -m "feat: tune_model + TuningResult one-call tuning interface

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Package exports

**Files:**
- Modify: `tests/test_model_selection.py`
- Modify: `src/gpbo/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_selection.py`:

```python
def test_public_api_exports():
    import gpbo

    assert gpbo.tune_model is tune_model
    assert gpbo.TuningResult is TuningResult
    assert gpbo.build_cv_objective is build_cv_objective
    assert gpbo.decode_parameters is decode_parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_selection.py::test_public_api_exports -v`
Expected: FAIL — `AttributeError: module 'gpbo' has no attribute 'tune_model'`

- [ ] **Step 3: Update `src/gpbo/__init__.py`**

Replace its full contents (current file is 11 lines) with:

```python
"""Gaussian Process regression and Bayesian optimization, from scratch."""

from gpbo.kernels import RBFKernel
from gpbo.gp import GaussianProcess
from gpbo.acquisition import expected_improvement
from gpbo.optimizer import BayesianOptimizer, OptimizationResult
from gpbo.model_selection import (
    TuningResult,
    build_cv_objective,
    decode_parameters,
    tune_model,
)

__all__ = [
    "RBFKernel",
    "GaussianProcess",
    "expected_improvement",
    "BayesianOptimizer",
    "OptimizationResult",
    "TuningResult",
    "build_cv_objective",
    "decode_parameters",
    "tune_model",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest`
Expected: `34 passed`

- [ ] **Step 5: Commit**

```bash
git add src/gpbo/__init__.py tests/test_model_selection.py
git commit -m "feat: export model_selection API from gpbo package

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Refactor the digits experiment + verification re-run

> **⚠ Long-running step in this task.** The experiment re-run takes ~10–25 min, which exceeds the 10-minute foreground Bash cap, and background processes started inside a task subagent die when that subagent's session ends (documented workflow lesson from this project). **Steps 3–5 must be executed from the main session** with a background Bash; a subagent may do Steps 1–2 and, after verification, Step 6.

**Files:**
- Modify: `experiments/hyperparameter_tuning.py:17` (imports), `:20` (imports), `:39-42` (objective)

- [ ] **Step 1: Apply the refactor edits**

Edit 1 — imports. Replace:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
```

with:

```python
from sklearn.model_selection import StratifiedKFold, train_test_split
```

Edit 2 — imports. Replace:

```python
from gpbo.optimizer import BayesianOptimizer
```

with:

```python
from gpbo.model_selection import build_cv_objective
from gpbo.optimizer import BayesianOptimizer
```

Edit 3 — the objective. Replace:

```python
def objective(params):
    a, b = params
    clf = SVC(C=10.0**a, gamma=10.0**b)
    return cross_val_score(clf, X_pool, y_pool, cv=CV, n_jobs=-1).mean()
```

with:

```python
def make_svc(params):
    return SVC(C=10.0**params["log10_C"], gamma=10.0**params["log10_gamma"])


objective = build_cv_objective(
    X_pool, y_pool, model_factory=make_svc,
    param_names=("log10_C", "log10_gamma"), cv=CV, n_jobs=-1,
)
```

Nothing else changes — not the module docstring, not `BOUNDS`/`N_SEEDS`/`N_INIT`/`N_ITER`/`CV`, not `random_search` (it calls the same `objective`, and the returned objective accepts its ndarray rows, as `landscape()`'s lists also decode fine), not the plotting, not the held-out block.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: `34 passed`

- [ ] **Step 3: Re-run the experiment (MAIN SESSION, background)**

Run (background Bash from the main session — completion re-invokes the session):

```bash
uv run python experiments/hyperparameter_tuning.py > /tmp/hp_rerun.log 2>&1
```

Expected duration: ~10–25 min (`data/digits_landscape.npz` cache exists, so no grid recomputation).

- [ ] **Step 4: Verify bit-identical results**

Run: `git status --porcelain figures/hp_comparison.png figures/hp_landscape.png data/digits_landscape.npz`
Expected: **empty output** (byte-identical figures, untouched cache).

Run: `tail -20 /tmp/hp_rerun.log`
Expected to contain exactly these lines (matching the README ground truth):

```
mean best after  5 evals:  BO 0.9837   RS 0.9820
mean best after 10 evals:  BO 0.9867   RS 0.9872
mean best after 25 evals:  BO 0.9887   RS 0.9877
BO best config: C=10^0.38, gamma=10^-0.91  -> held-out test accuracy 0.9889
RS best config: C=10^0.34, gamma=10^-0.60  -> held-out test accuracy 0.9889
```

- [ ] **Step 5: Gate**

If the `git status` output is non-empty or any printed number differs: **STOP. Do not commit.** Use superpowers:systematic-debugging to find the discrepancy (spec: "the refactor does not land unexplained"). Only proceed once the run is bit-identical.

- [ ] **Step 6: Commit**

```bash
git add experiments/hyperparameter_tuning.py
git commit -m "refactor: digits experiment consumes build_cv_objective

Objective construction now goes through the model_selection adapter;
verified by full re-run: hp_comparison.png and hp_landscape.png are
byte-identical (clean git status) and all printed checkpoints match
the README table (BO/RS 0.9837/0.9820, 0.9867/0.9872, 0.9887/0.9877;
held-out 0.9889 both).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Generalization demo

**Files:**
- Create: `experiments/generic_tuning_demo.py`
- Create (output): `figures/generic_tuning_demo.png`

- [ ] **Step 1: Write the demo**

Create `experiments/generic_tuning_demo.py`:

```python
"""Same tuning interface, different dataset and estimator.

The digits benchmark tunes an SVC through build_cv_objective; this script
tunes a scaled logistic regression on breast_cancer through tune_model —
same GP/EI machinery, zero optimizer changes. It exists to prove the
adapter generalizes, not to be another benchmark.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpbo.model_selection import tune_model

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

PARAM_SPACE = {"log10_C": (-4.0, 4.0)}
SEED = 0


def make_model(params):
    # The factory owns the 10**x transform: BO searches well-scaled log space,
    # sklearn receives the actual C. A Pipeline is just another estimator.
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=10.0 ** params["log10_C"], max_iter=1000)),
    ])


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    result = tune_model(
        X, y, model_factory=make_model, param_space=PARAM_SPACE, seed=SEED,
    )

    # Untuned baseline on the SAME folds tune_model used internally
    # (cv=5 int with seed=0 -> StratifiedKFold(5, shuffle=True, random_state=0)).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline = cross_val_score(make_model({"log10_C": 0.0}), X, y, cv=cv).mean()

    print(f"baseline C=1:  mean CV accuracy {baseline:.4f}")
    print(f"tuned  C=10^{result.best_params['log10_C']:.2f}:  "
          f"mean CV accuracy {result.best_cv_score:.4f}")

    best = result.optimization_result.best_so_far
    evals = np.arange(1, len(best) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evals, best, "C0", lw=2, label="best CV accuracy so far")
    ax.axhline(baseline, color="C1", ls="--", lw=1.5,
               label="untuned baseline (C=1)")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("mean 5-fold CV accuracy")
    ax.set_title("Tuning log₁₀C of a scaled logistic regression (breast_cancer)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "generic_tuning_demo.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'generic_tuning_demo.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python experiments/generic_tuning_demo.py`
Expected: finishes in well under a minute; prints a `baseline C=1:` line, a `tuned  C=10^...:` line with tuned ≥ baseline (both around 0.97–0.99), and `wrote .../figures/generic_tuning_demo.png`. Verify the PNG exists: `ls -la figures/generic_tuning_demo.png`.

- [ ] **Step 3: Commit (figure included — repo convention commits figures)**

```bash
git add experiments/generic_tuning_demo.py figures/generic_tuning_demo.png
git commit -m "feat: generic tuning demo — breast_cancer logistic regression

Proves tune_model works on a different dataset and estimator type
(a scaler+logreg Pipeline) with zero optimizer changes.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: README

**Files:**
- Modify: `README.md` (new section after the `hp_landscape.png` embed at line 66; "Correctness" test count at line 85; "How to run" lines 92 and 96–97)

- [ ] **Step 1: Insert the new section**

In `README.md`, immediately after the line `![Where BO samples](figures/hp_landscape.png)` (and its trailing blank line), and before `## What is implemented from scratch`, insert:

````markdown
## Reusable model tuning

Nothing in the optimizer is tied to digits or SVMs — the core maximizes an arbitrary `objective(x)` over a box. A small adapter (`src/gpbo/model_selection.py`) turns a dataset, an estimator factory, a parameter space, and a CV scheme into exactly such an objective, so the same GP/Expected-Improvement stack tunes any scikit-learn estimator:

```python
from gpbo import tune_model

def make_model(params):
    return Pipeline([("scale", StandardScaler()),
                     ("logreg", LogisticRegression(C=10.0 ** params["log10_C"], max_iter=1000))])

result = tune_model(X, y, model_factory=make_model,
                    param_space={"log10_C": (-4.0, 4.0)}, seed=0)
result.best_params      # {"log10_C": ...}
result.best_cv_score    # mean CV accuracy of the best configuration
```

The factory owns transforms like `C = 10**log10_C`, so BO searches a well-scaled space; the CV folds stay fixed for the whole run, which keeps the objective deterministic. The digits benchmark above runs through the same adapter (`build_cv_objective`), and `experiments/generic_tuning_demo.py` repeats the exercise on `breast_cancer` with a scaled logistic regression — different dataset, different estimator type, zero optimizer changes. Search dimensions are continuous floats only (no categorical or conditional parameters), and the caller prepares `X, y`.

````

- [ ] **Step 2: Update test counts and run commands**

In the "Correctness" section, replace:

```
$ uv run pytest
26 passed
```

with:

```
$ uv run pytest
34 passed
```

In "How to run", replace the line:

```bash
uv run pytest                                       # 26 passed
```

with:

```bash
uv run pytest                                       # 34 passed
```

and after the line ending `# BO vs RS on digits (~15–35 min first run)` add:

```bash
uv run python experiments/generic_tuning_demo.py    # reusable-tuning demo (<1 min)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README section on reusable model tuning

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification against the spec

**Files:** none modified (fix-forward only if a check fails)

- [ ] **Step 1: Full suite**

Run: `uv run pytest`
Expected: `34 passed`

- [ ] **Step 2: sklearn boundary intact**

Run: `grep -rln "sklearn" src/gpbo/`
Expected output — exactly one file:

```
src/gpbo/model_selection.py
```

- [ ] **Step 3: Nothing out-of-scope touched**

Run: `git status --porcelain && git diff HEAD --stat -- pyproject.toml docs/math-walkthrough.md`
Expected: empty working tree, no diff on either file (spec: both unchanged).

- [ ] **Step 4: Success-criteria walk**

Check each spec criterion and report the evidence: (1) 34 tests green; (2) Task 5's clean `git status` on figures + matching checkpoint prints; (3) demo runtime, prints, and PNG; (4) package imports + grep from Step 2; (5) README section present, pyproject/walkthrough untouched. Report results to the user; if any check fails, stop and fix before claiming completion (superpowers:verification-before-completion).
