import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gpbo.model_selection import (
    TuningResult,
    build_cv_objective,
    decode_parameters,
    tune_model,
)
from gpbo.optimizer import OptimizationResult

X_SMALL, Y_SMALL = make_classification(
    n_samples=80, n_features=6, n_informative=4, random_state=0
)


def _logreg_factory(params):
    return LogisticRegression(C=10.0 ** params["log10_C"], max_iter=200)


def test_decode_parameters_maps_names_in_order():
    params = decode_parameters(np.array([0.5, -1.5]), ("log10_C", "log10_gamma"))
    assert params == {"log10_C": 0.5, "log10_gamma": -1.5}
    assert all(type(v) is float for v in params.values())
    assert list(params) == ["log10_C", "log10_gamma"]   # dimension order preserved


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


def test_public_api_exports():
    import gpbo

    assert gpbo.tune_model is tune_model
    assert gpbo.TuningResult is TuningResult
    assert gpbo.build_cv_objective is build_cv_objective
    assert gpbo.decode_parameters is decode_parameters


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
