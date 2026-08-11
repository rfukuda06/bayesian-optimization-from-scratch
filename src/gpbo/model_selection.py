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

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gpbo.optimizer import BayesianOptimizer, OptimizationResult


def decode_parameters(x, param_names) -> dict:
    """Map the optimizer's vector x (d,) onto named parameters, in order.

    Dimension i of the search space is param_names[i]; for `tune_model` that
    order is the insertion order of `param_space` (guaranteed for dicts since
    Python 3.7). Values are coerced to plain floats so results print and
    serialize cleanly.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) != len(param_names):
        raise ValueError(f"x has {len(x)} values but param_names has {len(param_names)}")
    return {name: float(v) for name, v in zip(param_names, x)}


def build_cv_objective(X, y, model_factory, param_names, cv,
                       scoring=None, n_jobs=None):
    """Return objective(x) -> mean CV score, ready for BayesianOptimizer.

    `cv` is used exactly as given (anything `cross_val_score` accepts). Pass a
    splitter with a fixed random_state to make the objective deterministic —
    the module docstring explains why that matters for the GP. `scoring=None`
    delegates to the estimator's default scorer; any sklearn scoring string or
    scorer is passed straight through.
    """
    param_names = tuple(param_names)   # freeze order; caller mutations must not remap dimensions

    def objective(x):
        params = decode_parameters(x, param_names)
        model = model_factory(params)
        return cross_val_score(
            model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
        ).mean()

    return objective


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
    # covers plain and NumPy ints; bool is an int subclass we don't want to treat as one
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        cv = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=seed)
    objective = build_cv_objective(
        X, y, model_factory, names, cv, scoring=scoring, n_jobs=n_jobs
    )
    result = BayesianOptimizer(objective, bounds).run(n_init, n_iter, seed=seed)
    return TuningResult(
        best_params=decode_parameters(result.best_x, names),
        best_cv_score=result.best_y,
        optimization_result=result,
    )
