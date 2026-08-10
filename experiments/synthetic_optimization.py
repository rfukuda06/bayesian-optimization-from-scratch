"""BO on synthetic functions, replotted from OptimizationResult.history.

1D: f(x) = -sin(3x) - x^2 + 0.7x on [-1, 2] — per-iteration frames.
2D: Branin (negated for the maximize convention) — samples, GP mean, regret.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpbo.acquisition import expected_improvement
from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel
from gpbo.optimizer import BayesianOptimizer

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)


def gp_from_record(rec, bounds):
    """Rebuild the exact GP of one BO iteration from its history record:
    same observations, same fitted theta, same conditioning as the optimizer."""
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    X_unit = (rec.X - lo) / (hi - lo)
    y_std = rec.y.std()
    y_std = y_std if y_std > 1e-12 else 1.0
    y_s = (rec.y - rec.y.mean()) / y_std
    l, sf2, sn2 = rec.theta
    gp = GaussianProcess(RBFKernel(l, sf2), sn2)
    gp.fit(X_unit, y_s)
    return gp, rec.y.mean(), y_std, y_s.max()


# ---------------------------------------------------------------- 1D problem
def f1d(x):
    return -np.sin(3 * x[0]) - x[0] ** 2 + 0.7 * x[0]


def run_1d():
    bounds = [(-1.0, 2.0)]
    result = BayesianOptimizer(f1d, bounds).run(n_init=3, n_iter=12, seed=3)

    xs = np.linspace(-1, 2, 400)
    xs_unit = ((xs - (-1.0)) / 3.0)[:, None]
    f_true = np.array([f1d([x]) for x in xs])

    for i, rec in enumerate(result.history, start=1):
        gp, y_mean, y_std, y_best_s = gp_from_record(rec, bounds)
        mean_s, std_s = gp.predict(xs_unit)
        mean = y_mean + y_std * mean_s          # back to original units
        band = 2 * y_std * std_s
        ei = expected_improvement(mean_s, std_s, y_best_s)

        fig, (top, bot) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1]
        )
        top.plot(xs, f_true, "k--", lw=1, label="true objective")
        top.fill_between(xs, mean - band, mean + band, alpha=0.2, label="±2σ")
        top.plot(xs, mean, lw=2, label="GP mean")
        top.plot(rec.X, rec.y, "ko", ms=6, label="samples")
        top.axvline(rec.x_next[0], color="r", ls=":", label="next sample")
        top.set_title(f"Bayesian optimization, iteration {i}")
        top.legend(loc="lower left", fontsize=7)
        bot.plot(xs, ei, color="g")
        bot.axvline(rec.x_next[0], color="r", ls=":")
        bot.set_ylabel("EI (standardized)")
        bot.set_xlabel("x")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"bo_1d_iter_{i:02d}.png", dpi=120)
        plt.close(fig)

    print(f"1D: best f = {result.best_y:.4f} at x = {result.best_x[0]:.4f}")
    print(f"saved {len(result.history)} frames to figures/bo_1d_iter_*.png")


# ------------------------------------------------------------- Branin (2D)
BRANIN_MIN = 0.397887


def branin(x):
    x1, x2 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def run_branin():
    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    result = BayesianOptimizer(lambda x: -branin(x), bounds).run(
        n_init=5, n_iter=25, seed=0
    )

    g1, g2 = np.meshgrid(np.linspace(-5, 10, 120), np.linspace(0, 15, 120))
    Z = np.array(
        [branin([a, b]) for a, b in zip(g1.ravel(), g2.ravel())]
    ).reshape(g1.shape)

    # Samples over the true landscape (log-spaced levels: Branin spans decades)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cs = ax.contourf(g1, g2, Z, levels=np.logspace(-0.5, 2.5, 20), cmap="viridis")
    fig.colorbar(cs)
    ax.plot(result.X[:5, 0], result.X[:5, 1], "ws", ms=7, label="initial")
    ax.plot(result.X[5:, 0], result.X[5:, 1], "wo", ms=5, label="BO samples")
    for i, (x1, x2) in enumerate(result.X[5:], start=1):
        ax.annotate(str(i), (x1, x2), color="w", fontsize=6)
    ax.plot(*result.best_x, "r*", ms=15, label="best found")
    ax.set_title("Branin: sample sequence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_samples.png", dpi=150)
    plt.close(fig)

    # Final GP mean surface (negate back to minimization view for display)
    rec = result.history[-1]
    gp, y_mean, y_std, _ = gp_from_record(rec, bounds)
    U = np.column_stack(
        [(g1.ravel() + 5.0) / 15.0, g2.ravel() / 15.0]
    )
    mean_s, _ = gp.predict(U)
    surrogate = -(y_mean + y_std * mean_s).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cs = ax.contourf(g1, g2, surrogate, levels=20, cmap="viridis")
    fig.colorbar(cs)
    ax.plot(result.X[:, 0], result.X[:, 1], "wo", ms=4)
    ax.set_title("Final GP mean (surrogate of Branin)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_gp_mean.png", dpi=150)
    plt.close(fig)

    # Regret curve. best_so_far is the running max of -branin, so negating it
    # gives the running min of branin itself — already monotone.
    regret = -result.best_so_far - BRANIN_MIN
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(regret) + 1), np.maximum(regret, 1e-6))
    ax.set_xlabel("evaluation")
    ax.set_ylabel("|best − f*|")
    ax.set_title("Branin: simple regret")
    fig.tight_layout()
    fig.savefig(FIGDIR / "bo_branin_regret.png", dpi=150)
    plt.close(fig)

    print(f"Branin: best value {-result.best_y:.4f} (global min {BRANIN_MIN})")


if __name__ == "__main__":
    run_1d()
    run_branin()
