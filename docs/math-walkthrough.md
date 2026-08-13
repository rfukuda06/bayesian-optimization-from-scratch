# The math behind `gpbo`

The equations the library implements, with the derivations that matter. Each
section ends with a pointer to the code that implements the result.

**Notation.** Training inputs $X \in \mathbb{R}^{n \times d}$ with noisy targets
$y \in \mathbb{R}^n$; test inputs $X_\ast$ with latent values $f_\ast$. Gram
blocks: $K = k(X, X)$, $K_\ast = k(X, X_\ast)$, $K_{\ast\ast} = k(X_\ast, X_\ast)$.
The noisy training covariance is $K_y = K + \sigma_n^2 I$. Hyperparameters
$\theta = (\ell, \sigma_f^2, \sigma_n^2)$: length scale, signal variance, noise
variance. $\Phi$ and $\phi$ are the standard normal CDF and PDF.

---

## 1. The GP prior and the RBF kernel

A Gaussian process is a distribution over functions defined by one property: for
any finite set of inputs $x_1, \dots, x_n$, the vector $(f(x_1), \dots, f(x_n))$
is jointly Gaussian with mean $m(x_i)$ and covariance $k(x_i, x_j)$. Here
$m \equiv 0$ (§6 explains why standardization makes that reasonable). The
covariance function stands in for an infinite covariance matrix: any finite
slice ever computed is an ordinary multivariate Gaussian whose covariance matrix
is the Gram matrix $[k(x_i, x_j)]_{ij}$.

The library uses the squared-exponential (RBF) kernel

$$
k(x, x') = \sigma_f^2 \exp\left(-\frac{\lVert x - x' \rVert^2}{2\ell^2}\right).
$$

**Length scale $\ell$.** The correlation between function values at distance
$r$ is $e^{-r^2/2\ell^2}$: about $0.61$ at $r = \ell$, $0.14$ at $2\ell$, and
$0.01$ at $3\ell$. So $\ell$ is the distance over which function values stay
substantially correlated; beyond a few length scales the function is free to
wander, which is what small $\ell$ meaning "wiggly" amounts to.

**Signal variance $\sigma_f^2$.** Since $k(x, x) = \sigma_f^2$, the prior
marginal at every input is $\mathcal{N}(0, \sigma_f^2)$: sample paths live
mostly inside $\pm 2\sigma_f$. It multiplies the whole kernel, so it scales
covariances without changing any correlation.

**→ Code:** `src/gpbo/kernels.py`, `RBFKernel.__call__`, using the expansion
$\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\ a \cdot b$
(clipped at zero against floating-point cancellation).

---

## 2. Conditioning: the posterior equations

**The joint prior.** The model is $y_i = f(x_i) + \varepsilon_i$ with
$\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$ independent of $f$. The noisy
targets and the latent test values are both linear in jointly Gaussian
quantities, so they are jointly Gaussian, with
$\mathrm{Cov}(y, y) = K + \sigma_n^2 I$ (noise adds only on the diagonal) and
$\mathrm{Cov}(y, f_\ast) = K_\ast$:

$$
\begin{bmatrix} y \cr f_\ast \end{bmatrix}
\sim \mathcal{N}\left(0,\ 
\begin{bmatrix} K + \sigma_n^2 I & K_\ast \cr K_\ast^\top & K_{\ast\ast} \end{bmatrix}
\right).
$$

**The conditioning identity.** For jointly Gaussian vectors

$$
\begin{bmatrix} a \cr b \end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix} \mu_a \cr \mu_b \end{bmatrix},
\begin{bmatrix} A & C \cr C^\top & B \end{bmatrix}
\right),
\qquad
b \mid a \ \sim\  \mathcal{N}\big(\mu_b + C^\top A^{-1}(a - \mu_a),\ \  B - C^\top A^{-1} C\big).
$$

Proof sketch: define the residual $w = b - C^\top A^{-1} a$. Its
cross-covariance with $a$ is $C^\top - C^\top A^{-1} A = 0$, and $(w, a)$ is a
linear transform of a Gaussian, so $w$ is independent of $a$. Conditioning on
$a$ therefore leaves $w$ untouched, and $b = w + C^\top A^{-1} a$ is $w$ plus a
known constant: Gaussian with the stated mean, and covariance
$\mathrm{Cov}(w) = B - C^\top A^{-1} C$.

**Apply it** with $a = y$, $b = f_\ast$, $A = K + \sigma_n^2 I$,
$B = K_{\ast\ast}$, $C = K_\ast$:

$$
\boxed{\ 
\mu_\ast = K_\ast^\top (K + \sigma_n^2 I)^{-1} y,
\qquad
\Sigma_\ast = K_{\ast\ast} - K_\ast^\top (K + \sigma_n^2 I)^{-1} K_\ast.
\ }
$$

**Reading the equations.** With $\alpha = (K + \sigma_n^2 I)^{-1} y$ computed
once, the mean at any test point is $\mu_\ast(x_\ast) = \sum_i \alpha_i\ k(x_i, x_\ast)$:
a weighted sum of one kernel bump per training point. Far from all data the mean
reverts to $0$ and the variance to $\sigma_f^2$. The covariance is the prior
minus an explained term, and does not depend on $y$: under this model the error
bars are determined by where you observed, not what you saw. As
$\sigma_n^2 \to 0$ the mean becomes an interpolant; the equations predict the
latent $f_\ast$, and predicting a fresh noisy observation adds $\sigma_n^2$ back
to the variance.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess.predict` and
`GaussianProcess.sample_posterior`.

---

## 3. Cholesky and jitter

The posterior equations use $(K + \sigma_n^2 I)^{-1}$ three times; the library
never forms that inverse. $K_y$ is symmetric positive definite, so it factors as
$K_y = L L^\top$ with $L$ lower triangular (about $n^3/3$ flops), and every
inverse becomes a pair of $O(n^2)$ triangular solves: $\alpha = K_y^{-1} y$ via
$Lu = y$, then $L^\top \alpha = u$. Solving through the factorization is
backward-stable; forming the inverse explicitly costs about three times the
flops and squares the effect of the condition number, which matters because RBF
Gram matrices are routinely ill-conditioned.

For the predictive variance, one solve $V = L^{-1} K_\ast$ gives

$$
K_\ast^\top K_y^{-1} K_\ast = V^\top V,
$$

whose diagonal is a column-wise sum of squares. So each predictive variance is
$\sigma_f^2$ minus a sum of squares and cannot exceed the prior variance
(a `1e-12` clamp absorbs floating-point cancellation). The determinant needed in
§4 is free: $\log\lvert K_y \rvert = 2 \sum_i \log L_{ii}$.

**Jitter.** When $\sigma_n^2$ is tiny and inputs are close relative to $\ell$,
$K_y$ can lose positive definiteness numerically and Cholesky fails. The fix is
a small diagonal addition $\delta I$ before factorizing, which is a model change
rather than a hack:

$$
K + \sigma_n^2 I + \delta I = K + (\sigma_n^2 + \delta) I,
$$

the same GP with noise variance $\sigma_n^2 + \delta$. The code escalates
$\delta$ through $\lbrace 10^{-10}, \dots, 10^{-6} \rbrace$, warns on each step,
and raises only if the largest value still fails.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess._update_factorization` (jitter
ladder, cached $L$ and $\alpha$).

---

## 4. The log marginal likelihood

Marginalizing the latent function out of $y = f + \varepsilon$ needs no
integral: a sum of independent Gaussians is Gaussian, so
$y \mid X, \theta \sim \mathcal{N}(0, K_y)$. Taking the log of the density:

$$
\log p(y \mid X, \theta)
= \underbrace{-\tfrac{1}{2}\  y^\top K_y^{-1} y}_{\text{data fit}}
\ \underbrace{-\ \tfrac{1}{2} \log \lvert K_y \rvert}_{\text{complexity penalty}}
\ \underbrace{-\ \tfrac{n}{2} \log 2\pi}_{\text{constant}}.
$$

The data-fit term is the only one that sees the observed values; the determinant
term measures the volume of datasets the prior can generate. Because
$p(y \mid X, \theta)$ is a normalized density over all possible datasets, a
model flexible enough to explain anything spreads its probability mass thinly
and scores low at the actual data, while a model that concentrates mass near
datasets like the observed one scores high. Maximizing the LML therefore trades
fit against flexibility with no explicit regularizer; the optimum is where the
kernel's correlation structure captures the data's real smoothness.

`fit_hyperparameters` maximizes the LML over $\log \theta$ (log space keeps the
parameters positive and comparably scaled) with multi-start L-BFGS-B and
finite-difference gradients, inside fixed bounds stated for normalized inputs
and standardized $y$. The incoming $\theta$ is always kept as a candidate, so a
refit can never end worse than it started; a $\theta$ whose $K_y$ defeats the
jitter ladder receives a large finite penalty and the optimizer moves away.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess.log_marginal_likelihood`
(evaluates $-\tfrac12 y^\top \alpha - \sum_i \log L_{ii} - \tfrac n2 \log 2\pi$)
and `GaussianProcess.fit_hyperparameters`.

---

## 5. Expected Improvement

The incumbent is $y_{\text{best}}$, the best observed value (the library
maximizes; see §6). At a candidate $x$ the posterior is
$f \sim \mathcal{N}(\mu, \sigma^2)$. With a margin $\xi \ge 0$, define

$$
\mathrm{EI}(x) = \mathbb{E}\big[\max(f - y_{\text{best}} - \xi,\  0)\big].
$$

The $\max(\cdot, 0)$ is the point: outcomes below the bar cost nothing, so only
the upside tail counts.

**Derivation.** Assume $\sigma > 0$. Write $f = \mu + \sigma \epsilon$ with
$\epsilon \sim \mathcal{N}(0,1)$, and set

$$
I = \mu - y_{\text{best}} - \xi, \qquad z = \frac{I}{\sigma}.
$$

The integrand $\max(I + \sigma\epsilon, 0)$ is nonzero exactly when
$\epsilon > -z$, so

$$
\mathrm{EI}
= I \int_{-z}^{\infty} \phi(\epsilon)\  d\epsilon
\ +\  \sigma \int_{-z}^{\infty} \epsilon\  \phi(\epsilon)\  d\epsilon.
$$

The first integral is $1 - \Phi(-z) = \Phi(z)$. For the second, note
$\phi'(\epsilon) = -\epsilon\ \phi(\epsilon)$, so the integrand is an exact
derivative and the integral evaluates to $\phi(-z) = \phi(z)$. Therefore

$$
\boxed{\ \mathrm{EI}(x) = I\  \Phi(z) + \sigma\  \phi(z). \ }
$$

$I\ \Phi(z)$ is exploitation: the mean's headroom over the bar, weighted by the
probability of clearing it. $\sigma\ \phi(z)$ is exploration: a reward purely
for uncertainty, which keeps EI positive even where the mean is below the
incumbent. Larger $\xi$ discounts small mean advantages and tilts toward
exploration; the default $\xi = 0.01$ is in units of the standardized $y$ (§6).

**The $\sigma \to 0$ convention.** The true limit of the closed form is
$\max(\mu - y_{\text{best}} - \xi, 0)$. The implementation instead returns $0$
whenever $\sigma \le 10^{-12}$, regardless of the mean. This is deliberate:
inside the BO loop, numerically zero posterior uncertainty occurs at
already-evaluated points, and the proposal mechanism must never re-propose one
(there is nothing left to learn, and an argmax stuck on the incumbent stalls the
loop). A final clip to $[0, \infty)$ absorbs cancellation residue for very
negative $z$.

**→ Code:** `src/gpbo/acquisition.py`, `expected_improvement`.

---

## 6. The BO loop and its conventions

The loop: fit the GP to everything seen, refit hyperparameters, maximize EI over
the box, apply the duplicate guard, evaluate the objective, append, repeat. The
conventions around it:

- **Maximize-only.** One convention, one code path; minimizing $g$ is maximizing
  $-g$ at the call site. Supporting both internally would thread a sign through
  the incumbent, the improvement definition, and `best_so_far`.
- **Inputs normalized to $[0,1]^d$.** The kernel is isotropic (one $\ell$ for
  all dimensions), which is only meaningful if the coordinates share a scale. On
  the unit box, $\ell$ is a statement about fractions of each search range, and
  the fixed hyperparameter bounds and warm-start value are meaningful constants.
- **$y$ standardized every iteration.** The prior mean is zero; standardizing
  makes "revert to the prior" mean "revert to the average of what has been
  seen," keeps $\sigma_f^2$ near 1, and gives $\xi = 0.01$ its interpretation as
  1% of a standard deviation of the observations.
- **EI maximization.** In 1D, argmax over a dense 1000-point grid (deterministic,
  and identical to what the visualizations plot). In $d \ge 2$, 2048 seeded
  random candidates with L-BFGS-B refinement of the best few.
- **Duplicate guard.** A proposal within $10^{-6}$ of an existing point is
  replaced by a uniformly random point, so the loop cannot stall re-evaluating
  one location and gains exploration exactly when the model claims nothing is
  left to learn. `IterationRecord.ei_max` records the pre-guard argmax.
- **Warm starts.** The GP persists across iterations and the current $\theta$ is
  always kept as a candidate, so each refit starts from the previous answer.

**→ Code:** `src/gpbo/optimizer.py`, `BayesianOptimizer.run`,
`_maximize_ei_grid`, `_maximize_ei_candidates`, `_apply_duplicate_guard`.
