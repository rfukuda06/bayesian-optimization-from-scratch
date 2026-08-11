# Tuning Contract Brief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `tune_model` input/output contract explicit at every touchpoint — designed reprs, canonical docstring, README contract brief — with zero behavior changes.

**Architecture:** Three independent communication layers over an unchanged core: `__repr__` methods on the two result dataclasses (runtime), a NumPy-style docstring on `tune_model` (coding time), and a restructured README with a You-provide/You-get-back/Still-yours brief (reading time). Spec: `docs/superpowers/specs/2026-08-11-tuning-contract-brief-design.md`.

**Tech Stack:** Python 3.11, dataclasses, pytest, uv. No new dependencies.

**Hard constraint from the spec:** `git diff` over `src/` at the end must show only docstrings, comments, and repr code — no logic. All 34 existing tests must pass untouched.

---

## File Structure

- `src/gpbo/optimizer.py` — `OptimizationResult` gains `repr=False` + `__repr__` (single-line summary).
- `src/gpbo/model_selection.py` — `TuningResult` gains `repr=False` + `__repr__` (multi-line brief) and field comments; `tune_model` docstring rewritten NumPy-style.
- `tests/test_optimizer.py` — one exact-string repr test.
- `tests/test_model_selection.py` — one exact-string repr test + one real-run repr test.
- `README.md` — intro rewrite, Quick start section, contract brief, worked-example subsection, agent-prompt subsection, benchmark clause, test-count update.

Baseline commit before Task 1 is `73577df`. Final suite count: **37**.

---

### Task 1: OptimizationResult summary repr

**Files:**
- Modify: `src/gpbo/optimizer.py:70-77`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing test**

Add `OptimizationResult` to the existing first import block of `tests/test_optimizer.py`:

```python
from gpbo.optimizer import (
    OptimizationResult,
    _apply_duplicate_guard,
    _maximize_ei_candidates,
    _maximize_ei_grid,
)
```

Append at the end of the file:

```python
def test_optimization_result_repr_is_a_summary_not_an_array_dump():
    result = OptimizationResult(
        X=np.zeros((25, 2)),
        y=np.zeros(25),
        best_x=np.array([0.38, -0.91]),
        best_y=0.9876,
        best_so_far=np.zeros(25),
        history=[None] * 20,
    )
    assert repr(result) == (
        "OptimizationResult(n=25, d=2, best_y=0.9876; "
        "arrays: X, y, best_so_far; history: 20 iterations)"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_optimizer.py::test_optimization_result_repr_is_a_summary_not_an_array_dump -v`
Expected: FAIL — the default dataclass repr dumps the arrays, so the equality assertion fails.

- [ ] **Step 3: Implement the repr**

In `src/gpbo/optimizer.py`, replace the `OptimizationResult` dataclass (currently lines 70–77) with:

```python
@dataclass(repr=False)   # custom __repr__: the default would dump the arrays
class OptimizationResult:
    X: np.ndarray          # all evaluated points, original units, (n, d)
    y: np.ndarray          # (n,)
    best_x: np.ndarray
    best_y: float
    best_so_far: np.ndarray
    history: list          # list[IterationRecord], one per BO iteration

    def __repr__(self) -> str:
        n, d = self.X.shape
        return (
            f"OptimizationResult(n={n}, d={d}, best_y={self.best_y:.4f}; "
            f"arrays: X, y, best_so_far; history: {len(self.history)} iterations)"
        )
```

Nothing else in the file changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: all tests in the file PASS (existing 7 + the new one).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: `35 passed`

- [ ] **Step 6: Commit**

```bash
git add src/gpbo/optimizer.py tests/test_optimizer.py
git commit -m "feat: summary repr for OptimizationResult

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: TuningResult contract-brief repr

**Files:**
- Modify: `src/gpbo/model_selection.py:69-75`
- Test: `tests/test_model_selection.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_model_selection.py`, add below the existing imports:

```python
from gpbo.optimizer import OptimizationResult
```

Append at the end of the file:

```python
def _dummy_opt_result(n_evals, d, best_x, best_y, n_history):
    return OptimizationResult(
        X=np.zeros((n_evals, d)), y=np.zeros(n_evals), best_x=np.asarray(best_x),
        best_y=best_y, best_so_far=np.zeros(n_evals), history=[None] * n_history,
    )


def test_tuning_result_repr_exact_brief():
    result = TuningResult(
        best_params={"log10_C": -0.35},
        best_cv_score=0.9824,
        optimization_result=_dummy_opt_result(25, 1, [-0.35], 0.9824, 20),
    )
    assert repr(result) == "\n".join([
        "TuningResult  (25 evaluations)",
        "  best_params      {'log10_C': -0.35}",
        "  best_cv_score    0.9824",
        "  also available   .optimization_result.best_so_far  (best score per evaluation)",
        "                   .optimization_result.X, .y        (every config and score)",
        "  next step        model_factory(result.best_params).fit(X, y)",
    ])


def test_repr_of_real_result_shows_count_and_no_arrays():
    result = tune_model(
        X_SMALL, Y_SMALL, model_factory=_logreg_factory,
        param_space={"log10_C": (-2.0, 2.0)}, cv=3, n_init=3, n_iter=3, seed=0,
    )
    text = repr(result)
    assert text.splitlines()[0] == "TuningResult  (6 evaluations)"
    assert "array(" not in text
    assert len(text.splitlines()) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_selection.py -v -k repr`
Expected: both new tests FAIL (default dataclass repr).

- [ ] **Step 3: Implement the repr**

In `src/gpbo/model_selection.py`, replace the `TuningResult` dataclass (currently lines 69–75) with:

```python
@dataclass(repr=False)   # custom __repr__: the default would dump the arrays inside
class TuningResult:
    """What `tune_model` returns. Printing it shows the contract, not the arrays."""

    best_params: dict                        # {name: float} — feed back into your factory
    best_cv_score: float                     # mean CV score of best_params on the fixed folds
    optimization_result: OptimizationResult  # every evaluation + the best-so-far curve

    def __repr__(self) -> str:
        n = len(self.optimization_result.y)
        params = ", ".join(f"{k!r}: {v:.4g}" for k, v in self.best_params.items())
        return "\n".join([
            f"TuningResult  ({n} evaluations)",
            f"  best_params      {{{params}}}",
            f"  best_cv_score    {self.best_cv_score:.4f}",
            "  also available   .optimization_result.best_so_far  (best score per evaluation)",
            "                   .optimization_result.X, .y        (every config and score)",
            "  next step        model_factory(result.best_params).fit(X, y)",
        ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: all tests PASS (existing 8 + the two new ones).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: `37 passed`

- [ ] **Step 6: Commit**

```bash
git add src/gpbo/model_selection.py tests/test_model_selection.py
git commit -m "feat: TuningResult repr shows the contract brief

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Canonical tune_model docstring

**Files:**
- Modify: `src/gpbo/model_selection.py:80-94` (the `tune_model` docstring only — no code)

- [ ] **Step 1: Replace the docstring**

Replace the entire docstring of `tune_model` (the triple-quoted block, currently lines 80–94) with the following. Every semantic note in the old docstring is preserved; nothing about the function body changes.

```python
    """Tune `model_factory`'s hyperparameters over `param_space` with BO.

    Builds a deterministic cross-validation objective from (X, y), maximizes
    it with the GP/Expected-Improvement optimizer, and returns everything it
    learned. This function only tunes: printing, plotting, and the final fit
    are the caller-side lines shown in the example.

    Parameters
    ----------
    X, y : array-like
        Training data, already numeric. Cleaning, encoding, and feature
        scaling belong to the caller (put scalers inside the factory's
        Pipeline).
    model_factory : callable
        `model_factory(params: dict) -> unfitted estimator` (a Pipeline
        counts). Receives {name: float}; transforms like C = 10**log10_C
        live there, not here.
    param_space : dict
        Names to continuous (lo, hi) float bounds, e.g.
        {"log10_C": (-3.0, 3.0)}. Insertion order defines the optimizer's
        dimension order (guaranteed for dicts since Python 3.7).
    scoring : str or callable, optional
        Anything sklearn's cross_val_score accepts; None uses the
        estimator's default scorer. Scores are MAXIMIZED — minimize a loss
        via a negated scorer (e.g. scoring="neg_mean_squared_error").
    cv : int or splitter, default 5
        An int becomes StratifiedKFold(cv, shuffle=True, random_state=seed),
        a classification default whose folds are tied to `seed`. To hold
        folds fixed while varying `seed` (as experiments/hyperparameter_tuning.py
        does across trials), pass an explicit splitter; regression callers
        pass e.g. KFold. Folds are fixed for the whole run so the objective
        is deterministic.
    n_init, n_iter : int, default 5 and 20
        Mirror BayesianOptimizer.run: n_init random evaluations, then
        n_iter EI-guided ones.
    seed : int, default 0
        Seeds the folds (int `cv` only) and the optimizer. Same inputs and
        seed give identical results; a stochastic estimator must also pin
        its own random_state inside the factory.
    n_jobs : int, optional
        Passed through to cross_val_score for fold parallelism.

    Returns
    -------
    TuningResult
        .best_params          {name: float} — feed back into your factory
        .best_cv_score        best mean CV score found
        .optimization_result  every evaluation: .X, .y, .best_so_far
                              (convergence curve), .history

    Example
    -------
    >>> result = tune_model(X, y, make_model, {"log10_C": (-4.0, 4.0)})
    >>> model = make_model(result.best_params).fit(X, y)   # the final fit is yours
    >>> plt.plot(result.optimization_result.best_so_far)   # so is the plot
    """
```

(Doctests are not collected — pytest runs without `--doctest-modules` — so the example lines are documentation, not executed tests.)

- [ ] **Step 2: Verify it renders and nothing broke**

Run: `uv run python -c "import gpbo, pydoc; print(pydoc.render_doc(gpbo.tune_model))" | head -30`
Expected: the Parameters section renders; no import errors.

Run: `uv run pytest -q`
Expected: `37 passed`

- [ ] **Step 3: Commit**

```bash
git add src/gpbo/model_selection.py
git commit -m "docs: canonical NumPy-style docstring for tune_model

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: README restructure with contract brief

**Files:**
- Modify: `README.md`

Seven edits, top to bottom. Everything not mentioned stays untouched — in particular the division-of-labor bullets, the "Reading these honestly" paragraph, Limitations, and References.

- [ ] **Step 1: Rewrite the intro paragraph**

Replace the single dense paragraph directly under the `# Gaussian Processes + Bayesian Optimization from scratch` title with:

```markdown
Gaussian process regression and Bayesian optimization implemented from the math up in NumPy and SciPy. The whole stack is exposed as one reusable call, `tune_model`, which tunes the hyperparameters of any scikit-learn estimator on any dataset you provide. Every piece is validated — the GP against scikit-learn's, the acquisition function against a Monte-Carlo estimate, the optimizer against random search ([Correctness](#correctness) has the numbers). This is a learning project: the code is annotated at derivation grade and comes with a [full math walkthrough](docs/math-walkthrough.md).
```

(The `atol 1e-6` / `~1e-9` tolerances move out of the intro; the Correctness section already states them, so nothing is lost.)

- [ ] **Step 2: Insert a Quick start section**

Between the intro paragraph and `## The pipeline`, insert:

```markdown
## Quick start

```bash
uv sync
uv run pytest                                      # the full test suite, seconds
uv run python experiments/generic_tuning_demo.py   # end-to-end tuning demo, seconds
```
```

- [ ] **Step 3: Replace the "Fitting the final model is yours" paragraph with the contract brief**

In "Tune your model on your data", replace the paragraph
`Fitting the final model is yours: call your factory with `result.best_params` and train on your full training split — the library tunes, you deploy.`
with:

```markdown
The whole contract in one look:

```text
You provide   numeric X, y · a model_factory(params) -> estimator ·
              param_space bounds · optionally cv, seed, scoring, n_init/n_iter
You get back  TuningResult: .best_params, .best_cv_score,
              .optimization_result (all evaluations + best-so-far curve)
Still yours   the final fit — model_factory(result.best_params).fit(X, y) —
              plotting, and holding out a test set beforehand
```
```

- [ ] **Step 4: Relabel the breast-cancer demo as a worked example**

Replace the paragraph beginning
`` `experiments/generic_tuning_demo.py` is exactly this recipe run end to end on scikit-learn's `breast_cancer` dataset (569 tumor samples, 30 features) — ``
(keeping the figure embed line after it) with:

```markdown
### A worked example: breast cancer

The recipe above is not hypothetical — `experiments/generic_tuning_demo.py` is exactly that code run end to end on scikit-learn's built-in `breast_cancer` dataset (569 tumor samples, 30 numeric features, malignant/benign label). To tune your own data, copy that file and swap the loading lines and the factory. It finishes in seconds: on the committed run (seed 0), tuning lifts a scaled logistic regression from the untuned `C = 1` baseline's `0.9789` mean 5-fold CV accuracy to `0.9824` at `C = 10^-0.35`:
```

- [ ] **Step 5: Add the agent-prompt subsection**

After the `![Reusable tuning demo](figures/generic_tuning_demo.png)` line and before `## The benchmark: BO vs random search on digits`, insert:

```markdown
### Or hand it to your coding agent

Prefer not to write the recipe yourself? Paste this to a coding agent running inside a clone of this repo, filling in the blanks:

```text
Read the "Tune your model on your data" section of this repo's README, and use
experiments/generic_tuning_demo.py as the template.

Write and run a script like that demo, but for my data: it lives at <PATH> and
the target column is <NAME> — get it into the numeric X, y the library expects.
Tune a <MODEL — or pick a sensible scikit-learn model for me>, searching
<KNOBS AND RANGES — or pick 1–3 standard knobs, log-scaled where sensible>.
At the end, refit the best model on all my data so I can use it.
```
```

- [ ] **Step 6: Benchmark clause + test count**

In the benchmark section's opening paragraph, replace
`all through the same `build_cv_objective` adapter`
with
`all driven through `build_cv_objective`, the machinery layer underneath `tune_model``.

In the Correctness section's fenced block, replace `34 passed` with `37 passed`.

- [ ] **Step 7: Verify structure and suite**

Run: `grep -n "^## \|^### " README.md`
Expected order: Quick start · The pipeline · Tune your model on your data · A worked example: breast cancer · Or hand it to your coding agent · The benchmark… · Under the hood… · GP regression · 1D Bayesian optimization · 2D Bayesian optimization (Branin) · What is implemented from scratch · Correctness · How to run · Limitations · References.

Run: `uv run pytest -q`
Expected: `37 passed`

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: README restructure — quick start, contract brief, worked example, agent prompt

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Final verification and push

**Files:** none created or modified (verification only).

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: `37 passed`

- [ ] **Step 2: No-logic-change check (spec success criterion)**

Run: `git diff 73577df..HEAD -- src/`
Expected: every changed line in `src/` is a docstring, comment, `repr=False`, or `__repr__` body. Any changed line that executes tuning logic is a plan violation — stop and report.

- [ ] **Step 3: Behavior tripwire — demo reruns identically**

Run: `uv run python experiments/generic_tuning_demo.py`
Expected output includes `baseline C=1:  mean CV accuracy 0.9789` and the tuned score `0.9824` — identical to the README numbers.

Run: `git status --porcelain`
Expected: empty. The demo rewrote `figures/generic_tuning_demo.png` byte-identically; a dirty figure means behavior drifted — stop and report.

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-review notes (already applied)

- Spec coverage: brief → Task 4 Step 3; docstring → Task 3; reprs → Tasks 1–2; restructure items → Task 4 Steps 1, 2, 4, 5, 6; tests → Tasks 1–2 (three new, suite 37); success criteria → Task 5.
- Repr format strings in Tasks 1–2 match the spec sketch exactly (`.4g` params, `.4f` scores, `len(optimization_result.y)` count).
- `test_public_api_exports` is untouched — the spec forbids export changes.
- Float display values chosen to render exactly under `.4f`/`.4g` (0.9876, 0.9824, -0.35).
