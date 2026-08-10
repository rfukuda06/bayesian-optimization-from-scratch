"""1D GP visualization: prior samples, posterior with 3 then 8 observations.

Also prints (a) the sklearn agreement check at fixed hyperparameters,
(b) our fitted LML vs sklearn's fitted LML, and (c) the non-blocking
hyperparameter-recovery sanity experiment from the spec.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

NOISE_STD = 0.2


def f(x):
    return x * np.sin(x)


def plot_panel(ax, gp, X_obs, y_obs, title, rng):
    xs = np.linspace(0, 10, 400)[:, None]
    mean, std = gp.predict(xs)
    ax.plot(xs, f(xs.ravel()), "k--", lw=1, label="true f(x) = x sin(x)")
    ax.fill_between(
        xs.ravel(), mean - 2 * std, mean + 2 * std, alpha=0.2, label="±2σ"
    )
    ax.plot(xs, mean, lw=2, label="posterior mean")
    for s in gp.sample_posterior(xs, 4, rng=rng):
        ax.plot(xs, s, lw=0.6, alpha=0.6)
    ax.plot(X_obs, y_obs, "ko", ms=6, label="observations")
    ax.set_title(title)
    ax.set_xlim(0, 10)


def main():
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    # (a) Prior: zero mean, bands at ±2 sqrt(signal_variance), sampled functions
    # drawn directly from the kernel matrix — no fit involved.
    xs = np.linspace(0, 10, 400)[:, None]
    prior_kernel = RBFKernel(length_scale=1.0, signal_variance=4.0)
    K = prior_kernel(xs, xs) + 1e-10 * np.eye(len(xs))
    L = np.linalg.cholesky(K)
    axes[0].plot(xs, L @ rng.standard_normal((len(xs), 4)), lw=0.8)
    axes[0].fill_between(xs.ravel(), -2 * 2.0, 2 * 2.0, alpha=0.15)
    axes[0].axhline(0, color="C0", lw=2)
    axes[0].set_title("(a) GP prior: mean 0, ±2σ_f, 4 sampled functions")

    # (b), (c) Posterior after 3 and 8 noisy observations, hyperparameters fitted.
    X_all = rng.uniform(0.5, 9.5, size=(8, 1))
    y_all = f(X_all.ravel()) + NOISE_STD * rng.standard_normal(8)
    for ax, n, label in [(axes[1], 3, "(b)"), (axes[2], 8, "(c)")]:
        gp = GaussianProcess(RBFKernel(1.0, 4.0), NOISE_STD**2)
        gp.fit(X_all[:n], y_all[:n])
        gp.fit_hyperparameters(rng=np.random.default_rng(1))
        plot_panel(
            ax, gp, X_all[:n], y_all[:n], f"{label} posterior, {n} observations", rng
        )
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "gp_demo.png", dpi=150)
    print(f"saved {FIGDIR / 'gp_demo.png'}")

    # --- sklearn agreement at fixed theta -------------------------------------
    ours = GaussianProcess(RBFKernel(1.5, 4.0), 0.04)
    ours.fit(X_all, y_all)
    sk = GaussianProcessRegressor(
        kernel=ConstantKernel(4.0, "fixed") * RBF(1.5, "fixed"),
        alpha=0.04,
        optimizer=None,
        normalize_y=False,
    ).fit(X_all, y_all)
    xs_test = np.linspace(0, 10, 100)[:, None]
    m1, s1 = ours.predict(xs_test)
    m2, s2 = sk.predict(xs_test, return_std=True)
    print(f"fixed-theta max|Δmean| = {np.abs(m1 - m2).max():.2e}")
    print(f"fixed-theta max|Δstd|  = {np.abs(s1 - s2).max():.2e}")

    # --- fitted LML: ours vs sklearn's optimizer ------------------------------
    fitted = GaussianProcess(RBFKernel(1.0, 1.0), 0.1)
    fitted.fit(X_all, y_all)
    fitted.fit_hyperparameters(rng=np.random.default_rng(2))
    sk_opt = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0), alpha=0.04, n_restarts_optimizer=5
    ).fit(X_all, y_all)
    print(f"our fitted LML     = {fitted.log_marginal_likelihood():.4f}")
    print(f"sklearn fitted LML = {sk_opt.log_marginal_likelihood():.4f}")
    print("(not directly comparable models: sklearn's noise here is fixed alpha)")

    # --- non-blocking recovery sanity experiment (spec §6.1 / §7) -------------
    true_l = 1.0
    gen = RBFKernel(true_l, 1.0)
    Xg = np.linspace(0, 10, 40)[:, None]
    Kg = gen(Xg, Xg) + 1e-8 * np.eye(40)
    yg = np.linalg.cholesky(Kg) @ np.random.default_rng(5).standard_normal(40)
    rec = GaussianProcess(RBFKernel(3.0, 0.5), 1e-4)
    rec.fit(Xg, yg)
    rec.fit_hyperparameters(rng=np.random.default_rng(6))
    print(
        f"recovery sanity: true length_scale={true_l}, "
        f"recovered={rec.kernel.length_scale:.3f} (informational only)"
    )


if __name__ == "__main__":
    main()
