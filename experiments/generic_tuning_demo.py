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
