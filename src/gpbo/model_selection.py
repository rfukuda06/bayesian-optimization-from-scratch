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
from sklearn.model_selection import cross_val_score


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
