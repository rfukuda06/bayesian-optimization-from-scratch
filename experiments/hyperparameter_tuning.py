"""BO vs random search tuning SVC(C, gamma) on digits (spec §6.3).

Search space: a = log10(C) in [-3, 3], b = log10(gamma) in [-5, 1].
Objective: mean 5-fold stratified CV accuracy on the 80% pool.
Budget: 25 evaluations per method per trial; 10 seeded trials each.
Held-out 20% test set is touched exactly once per method at the end.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

from gpbo.optimizer import BayesianOptimizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGDIR, DATADIR = ROOT / "figures", ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)

BOUNDS = [(-3.0, 3.0), (-5.0, 1.0)]
N_SEEDS = 10
N_INIT, N_ITER = 5, 20          # 25 evaluations total
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

digits = load_digits()
X_pool, X_test, y_pool, y_test = train_test_split(
    digits.data / 16.0, digits.target, test_size=0.2, stratify=digits.target,
    random_state=0,
)


def objective(params):
    a, b = params
    clf = SVC(C=10.0**a, gamma=10.0**b)
    return cross_val_score(clf, X_pool, y_pool, cv=CV, n_jobs=-1).mean()


def random_search(n_evals, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    params = lo + rng.uniform(size=(n_evals, 2)) * (hi - lo)
    y = np.array([objective(p) for p in params])
    return params, y


def landscape():
    """20x20 ground-truth CV-accuracy grid, cached (~10 min first run)."""
    cache = DATADIR / "digits_landscape.npz"
    if cache.exists():
        d = np.load(cache)
        return d["A"], d["B"], d["Z"]
    A, B = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-5, 1, 20))
    Z = np.array(
        [objective([a, b]) for a, b in zip(A.ravel(), B.ravel())]
    ).reshape(A.shape)
    np.savez(cache, A=A, B=B, Z=Z)
    return A, B, Z


def main():
    bo_curves, rs_curves = [], []
    bo_best, rs_best = (-np.inf, None), (-np.inf, None)
    for seed in range(N_SEEDS):
        r = BayesianOptimizer(objective, BOUNDS).run(N_INIT, N_ITER, seed=seed)
        if seed == 0:
            r0 = r  # kept for the landscape plot below
        bo_curves.append(r.best_so_far)
        if r.best_y > bo_best[0]:
            bo_best = (r.best_y, r.best_x)

        params, y = random_search(N_INIT + N_ITER, seed=100 + seed)
        rs_curves.append(np.maximum.accumulate(y))
        if y.max() > rs_best[0]:
            rs_best = (y.max(), params[np.argmax(y)])
        print(f"seed {seed}: BO best {bo_curves[-1][-1]:.4f}  "
              f"RS best {rs_curves[-1][-1]:.4f}")

    bo_curves, rs_curves = np.array(bo_curves), np.array(rs_curves)
    evals = np.arange(1, N_INIT + N_ITER + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curves, label, color in [
        (bo_curves, "Bayesian optimization", "C0"),
        (rs_curves, "Random search", "C1"),
    ]:
        m, s = curves.mean(axis=0), curves.std(axis=0)
        ax.plot(evals, m, color=color, lw=2, label=label)
        ax.fill_between(evals, m - s, m + s, color=color, alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best 5-fold CV accuracy so far")
    ax.set_title(f"SVC(C, γ) on digits — mean ± std over {N_SEEDS} seeds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "hp_comparison.png", dpi=150)
    plt.close(fig)

    for n in (5, 10, 25):
        print(f"mean best after {n:2d} evals:  "
              f"BO {bo_curves[:, n - 1].mean():.4f}   "
              f"RS {rs_curves[:, n - 1].mean():.4f}")

    # BO sample placement of the seed-0 run (saved above) over the landscape
    A, B, Z = landscape()
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(A, B, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, label="CV accuracy")
    ax.plot(r0.X[:N_INIT, 0], r0.X[:N_INIT, 1], "ws", ms=7, label="initial")
    ax.plot(r0.X[N_INIT:, 0], r0.X[N_INIT:, 1], "wo", ms=5, label="BO samples")
    ax.plot(*r0.best_x, "r*", ms=15, label="best")
    ax.set_xlabel("log10 C")
    ax.set_ylabel("log10 gamma")
    ax.set_title("Where BO samples (seed 0)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hp_landscape.png", dpi=150)
    plt.close(fig)

    # Held-out test: one shot per method with its overall best config
    for name, (_, best_params) in [("BO", bo_best), ("RS", rs_best)]:
        a, b = best_params
        clf = SVC(C=10.0**a, gamma=10.0**b).fit(X_pool, y_pool)
        print(f"{name} best config: C=10^{a:.2f}, gamma=10^{b:.2f}  "
              f"-> held-out test accuracy {clf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
