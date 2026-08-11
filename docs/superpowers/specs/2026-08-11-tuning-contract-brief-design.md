# Design: the tuning contract brief

**Date:** 2026-08-11
**Status:** approved in discussion; implementation pending

## Context

`tune_model` deliberately returns data and stays silent: printing results,
plotting the convergence curve, and fitting the final model are caller-side
steps. That boundary is sound library design, but the project's own author
found the usage story unclear — nothing tells a user, at the moment they hold
a `TuningResult`, what they gave the library, what they got back, and what is
still theirs to do. A full-service redesign (auto-print, auto-plot,
auto-refit) was designed, weighed, and **rejected**: it would trade the
library discipline this portfolio project showcases for convenience, and the
clarity problem can be solved by communication instead of behavior.

A concrete wart motivates the runtime piece: `print(result)` today falls
through to the default dataclass repr, which dumps `OptimizationResult`'s
NumPy arrays — a wall of numbers.

## Goal

Make the input/output contract of `tune_model` explicit at every moment a
user meets it, **without changing any behavior**:

1. **Before calling** — a contract brief in the README.
2. **While coding** — the docstring as the canonical reference.
3. **After running** — a designed repr on the result objects.

## Non-goals

- No behavior change to `tune_model` (no printing, no plotting, no refit, no
  `.model` field, no `verbose` flag).
- No API or export changes; `build_cv_objective` and `decode_parameters`
  keep their current visibility.
- No `summary()` method — the repr is the single runtime mechanism.
- No changes to the benchmark or demo scripts' behavior.

## Design

### 1. README contract brief

A three-row block in the "Tune your model on your data" section, between the
recipe snippet and the division-of-labor bullets:

```
You provide   numeric X, y · a model_factory(params) -> estimator ·
              param_space bounds · optionally cv, seed, scoring, n_init/n_iter
You get back  TuningResult: .best_params, .best_cv_score,
              .optimization_result (all evaluations + best-so-far curve)
Still yours   the final fit — model_factory(result.best_params).fit(X, y) —
              plotting, and holding out a test set beforehand
```

Rendered as a small table or fenced block (implementer's choice; must stay
under ~8 lines). The wording mirrors the docstring so the two cannot drift
apart in substance.

### 2. Docstring canonicalization

`tune_model`'s docstring becomes the canonical contract, NumPy style:
Parameters (all ten, with the factory and param-space contracts spelled
out), Returns (each `TuningResult` field and what it is for), and a short
Example ending with the two caller-side lines (final fit, one-line plot).
`TuningResult` and `OptimizationResult` get one-line field docs. No content
may contradict the README brief.

### 3. Designed reprs

`TuningResult` and `OptimizationResult` become `@dataclass(repr=False)` with
custom `__repr__`s. Target shape (exact strings pinned by tests):

```
TuningResult  (25 evaluations)
  best_params      {'log10_C': -0.35}
  best_cv_score    0.9824
  also available   .optimization_result.best_so_far  (best score per evaluation)
                   .optimization_result.X, .y        (every config and score)
  next step        model_factory(result.best_params).fit(X, y)
```

Formatting rules:

- Header count is `len(optimization_result.y)`.
- `best_params` values formatted `{v:.4g}`; `best_cv_score` formatted `.4f`.
- Two aligned columns; the `next step` line names `model_factory`, the
  parameter the user passed, and `result`, the conventional variable name.
- `OptimizationResult.__repr__` is a single line — element count, `best_y`,
  input dimension, and the names of the array/history fields — so the object
  the brief points into does not itself dump arrays.
- Reprs never raise: they assume only that the dataclass fields exist, and
  they must render correctly for hand-constructed instances (as the tests
  construct them) as well as real tuning output.

## Accompanying README restructure (already agreed, same delivery)

Riding in the same implementation pass, previously approved in discussion:

- Intro rewritten to 3–4 plain sentences; numeric tolerances move to the
  Correctness section (deduplicating the current repetition).
- A 3-line quick start (`uv sync`, `uv run pytest`,
  `uv run python experiments/generic_tuning_demo.py`) directly after the intro.
- The breast-cancer demo relabeled as an explicit subsection ("A worked
  example…") with a framing sentence naming the dataset and stating that the
  demo is the recipe run end to end.
- An agent-prompt block closing the usage section (approved wording: reads
  the README section and the demo as template; three blanks — data path +
  target column, model, knobs/ranges — each with an agent-picks fallback;
  one final-refit clause; assumes the repo is already cloned).
- One clause in the benchmark section noting it drives `build_cv_objective`
  (the machinery under `tune_model`) directly.

## Testing

- Exact-string repr tests for `TuningResult` and `OptimizationResult` on
  hand-constructed instances with known values (no tuning run needed).
- One integration assertion that a real (tiny, seeded) `tune_model` result
  reprs without error and contains its evaluation count.
- Existing 34 tests unchanged — any failure there means behavior drifted,
  which this design forbids. Suite lands at roughly 37.

## Success criteria

- `print(result)` shows the brief above, never an array dump.
- README usage section reads: recipe → contract brief → bullets → worked
  example → agent prompt, with the lighter intro and quick start in place.
- `help(gpbo.tune_model)` alone is sufficient to use the library correctly.
- `git diff` over `src/` shows only docstrings and repr code — no logic.
