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
