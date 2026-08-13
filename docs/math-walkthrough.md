# The math behind `gpbo`, worked end to end

Every equation this library implements, derived from first principles. Each section
ends with a pointer to the code that implements the result. The target reader is
someone rehearsing explanations out loud: every nontrivial step is shown, and every
result gets a plain-language reading.

**Notation used throughout.** Inputs $x \in \mathbb{R}^d$. Training inputs
$X \in \mathbb{R}^{n \times d}$ with noisy targets $y \in \mathbb{R}^n$; test inputs
$X_\ast \in \mathbb{R}^{m \times d}$ with latent function values $f_\ast \in \mathbb{R}^m$.
Kernel Gram blocks: $K = k(X, X)$ is $n \times n$, $K_\ast = k(X, X_\ast)$ is
$n \times m$, and $K_{\ast\ast} = k(X_\ast, X_\ast)$ is $m \times m$. The noisy
training covariance is $K_y = K + \sigma_n^2 I$. Hyperparameters
$\theta = (\ell, \sigma_f^2, \sigma_n^2)$: length scale, signal variance, noise
variance. $\Phi$ and $\phi$ are the standard normal CDF and PDF.

---

## 1. From Gaussians to function priors

**Univariate to multivariate.** A univariate Gaussian
$x \sim \mathcal{N}(\mu, \sigma^2)$ has density
$p(x) = (2\pi\sigma^2)^{-1/2} \exp\left(-\tfrac{(x-\mu)^2}{2\sigma^2}\right)$.
The multivariate version replaces the variance with a covariance **matrix**
$\Sigma \in \mathbb{R}^{n \times n}$:

$$
p(\mathbf{x}) = (2\pi)^{-n/2} \lvert\Sigma\rvert^{-1/2}
\exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right).
$$

The entry $\Sigma_{ij} = \mathrm{Cov}(x_i, x_j)$ is the whole story: it says how
much learning the value of $x_i$ should move your belief about $x_j$. Diagonal
entries are marginal variances; off-diagonal entries encode coupling. A Gaussian
with a rich covariance matrix is not $n$ separate beliefs — it is one joint belief
where components inform each other.

**The jump to functions.** A Gaussian process is what you get when you push $n \to \infty$:
a distribution over entire functions $f$, defined by the property that for **any**
finite set of inputs $x_1, \dots, x_n$, the vector $(f(x_1), \dots, f(x_n))$ is jointly
Gaussian with

$$
\mathbb{E}[f(x_i)] = m(x_i), \qquad \mathrm{Cov}(f(x_i), f(x_j)) = k(x_i, x_j).
$$

In this library $m \equiv 0$ (see §6 for why that is honest after standardization).
The trick that makes the infinite-dimensional object manageable: instead of writing
down an infinite covariance matrix, you write down a covariance **function** $k$ that
can manufacture the covariance between any pair of function values on demand. Any
finite slice you ever compute with is an ordinary multivariate Gaussian whose
covariance matrix is the Gram matrix $[k(x_i, x_j)]_{ij}$.

For this to be self-consistent (Kolmogorov consistency: marginalizing a bigger slice
must give the smaller slice), every Gram matrix must be a valid covariance matrix,
i.e. symmetric positive semidefinite. The RBF kernel satisfies this because it is an
inner product of feature maps, $k(x, x') = \langle \varphi(x), \varphi(x') \rangle$
(for RBF the feature space is infinite-dimensional), so for any $v \in \mathbb{R}^n$:

$$
v^\top K v = \Big\lVert \textstyle\sum_i v_i\  \varphi(x_i) \Big\rVert^2 \ge 0.
$$

**The RBF kernel and its two knobs.** This repo uses the squared-exponential (RBF)
kernel:

$$
k(x, x') = \sigma_f^2 \exp\left(-\frac{\lVert x - x' \rVert^2}{2\ell^2}\right).
$$

*Length scale $\ell$ controls wiggliness.* The correlation between two function
values at distance $r = \lVert x - x' \rVert$ is

$$
\mathrm{corr}\big(f(x), f(x')\big) = \frac{k(x,x')}{\sigma_f^2} = e^{-r^2/2\ell^2},
$$

which is $\approx 0.61$ at $r = \ell$, $\approx 0.14$ at $r = 2\ell$, and $\approx 0.01$
at $r = 3\ell$. So $\ell$ is literally "the distance over which function values stay
substantially correlated": beyond a few length scales, knowing $f(x)$ tells you almost
nothing about $f(x')$, so the function is free to wander — that freedom is wiggliness.

To make this quantitative, look at the derivative process in 1D. For a stationary
kernel $k(x, x') = k(\tau)$ with $\tau = x - x'$, differentiation passes through the
covariance (the derivative of a Gaussian process is again Gaussian, with covariance
obtained by differentiating $k$ once in each argument):

$$
\mathrm{Cov}\big(f'(x), f'(x')\big)
= \frac{\partial^2}{\partial x\  \partial x'} k(x - x')
= -k''(\tau).
$$

For the RBF kernel, $k(\tau) = \sigma_f^2 e^{-\tau^2/2\ell^2}$, so
$k'(\tau) = -\sigma_f^2 \frac{\tau}{\ell^2} e^{-\tau^2/2\ell^2}$ and
$k''(\tau) = \sigma_f^2 \left(\frac{\tau^2}{\ell^4} - \frac{1}{\ell^2}\right) e^{-\tau^2/2\ell^2}$.
Evaluating at $\tau = 0$:

$$
\mathrm{Var}\big(f'(x)\big) = -k''(0) = \frac{\sigma_f^2}{\ell^2}.
$$

The typical slope of a sample path is $\sigma_f / \ell$: halve the length scale and
the typical slope doubles. That is the precise sense in which small $\ell$ means
wiggly.

*Signal variance $\sigma_f^2$ controls amplitude.* Since $k(x, x) = \sigma_f^2$, the
prior marginal at every input is $f(x) \sim \mathcal{N}(0, \sigma_f^2)$: sample paths
live mostly inside the band $\pm 2\sigma_f$. And because $\sigma_f^2$ multiplies the
whole kernel, it scales every covariance without changing any correlation — it is a
pure vertical-stretch knob, orthogonal to the shape knob $\ell$.

**→ Code:** `src/gpbo/kernels.py`, `RBFKernel.__call__` — builds the Gram matrix for
any two point sets using the expansion
$\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\  a \cdot b$
(clipped at zero because floating-point cancellation can make near-identical points
produce tiny negative squared distances); `_as_2d` coerces 1D inputs to $(n, 1)$.

---

## 2. Conditioning: the posterior equations

**The joint prior.** The model is $y_i = f(x_i) + \varepsilon_i$ with
$\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$ independent of $f$. Stack the noisy
training targets $y$ and the latent test values $f_\ast$. Both are linear in jointly
Gaussian quantities ($f$ at various inputs, plus independent Gaussian noise), so they
are jointly Gaussian. The blocks: $\mathrm{Cov}(y, y) = K + \sigma_n^2 I$ (noise
adds only on the diagonal, and only to observed values), and
$\mathrm{Cov}(y, f_\ast) = \mathrm{Cov}(f + \varepsilon, f_\ast) = K_\ast$
because the noise is independent of everything. Hence

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
$$

the conditional is

$$
b \mid a \ \sim\  \mathcal{N}\big(\mu_b + C^\top A^{-1}(a - \mu_a),\ \  B - C^\top A^{-1} C\big).
$$

*Why this is true* (worked, no completing-the-square slog). Define the residual
$w = b - C^\top A^{-1} a$, i.e. what is left of $b$ after subtracting its best linear
predictor from $a$. Check its cross-covariance with $a$:

$$
\mathrm{Cov}(w, a) = \mathrm{Cov}(b, a) - C^\top A^{-1} \mathrm{Cov}(a, a)
= C^\top - C^\top A^{-1} A = 0.
$$

The pair $(w, a)$ is a linear transform of the jointly Gaussian $(a, b)$, hence
itself jointly Gaussian — and for jointly Gaussian variables, zero covariance implies
independence. So conditioning on $a$ does not change the distribution of $w$ at all. Its (unconditional) moments:
$\mathbb{E}[w] = \mu_b - C^\top A^{-1} \mu_a$, and with $M = C^\top A^{-1}$,

$$
\mathrm{Cov}(w)
= B - \mathrm{Cov}(b,a) M^\top - M \mathrm{Cov}(a,b) + M A M^\top
= B - 2\ C^\top A^{-1} C + C^\top A^{-1} C
= B - C^\top A^{-1} C.
$$

Now write $b = w + M a$. Given $a$, the term $Ma$ is a known constant, so $b \mid a$
is $w$ shifted by $Ma$ — Gaussian with mean $\mu_b + M(a - \mu_a)$ and covariance
$\mathrm{Cov}(w)$. That is the identity.

**Apply it.** Set $a = y$, $b = f_\ast$, $\mu_a = \mu_b = 0$, $A = K + \sigma_n^2 I$,
$B = K_{\ast\ast}$, $C = K_\ast$:

$$
\boxed{\ 
\mu_\ast = K_\ast^\top (K + \sigma_n^2 I)^{-1} y,
\qquad
\Sigma_\ast = K_{\ast\ast} - K_\ast^\top (K + \sigma_n^2 I)^{-1} K_\ast.
\ }
$$

Dimensions check out: $K_\ast^\top$ is $m \times n$, the inverse is $n \times n$, $y$ is
$n$, so $\mu_\ast \in \mathbb{R}^m$; $\Sigma_\ast$ is $m \times m$.

**How to read these.**

- *The mean is a weighted sum of kernel bumps.* Define
  $\alpha = (K + \sigma_n^2 I)^{-1} y \in \mathbb{R}^n$ once; then for any test point,
  $\mu_\ast(x_\ast) = \sum_{i=1}^n \alpha_i\  k(x_i, x_\ast)$. The posterior mean is a
  linear combination of one kernel function centered at each training point — data
  points near $x_\ast$ dominate, and points beyond a few length scales contribute
  nothing. Far from all data, $K_\ast \to 0$, so the mean reverts to the prior mean 0
  and the variance reverts to $\sigma_f^2$.
- *The covariance is prior minus explained.* $\Sigma_\ast$ starts from the prior
  uncertainty $K_{\ast\ast}$ and subtracts a positive semidefinite term measuring how
  much the training locations pin down the test values. Notably, $\Sigma_\ast$ does
  **not** depend on $y$ — only on *where* you observed, not *what* you saw. That is a
  real (and strong) property of the homoscedastic Gaussian model: you know your error
  bars before running the experiments.
- *Noise in, noise out.* $\sigma_n^2$ appears inside the inverse: with noise, the
  posterior mean is a smoother that need not pass through the data; as
  $\sigma_n^2 \to 0$ it becomes an interpolant. The equations above predict the latent
  $f_\ast$; to predict a fresh noisy *observation* $y_\ast$, add $\sigma_n^2$ back to
  the variance.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess.predict` — computes
$\mu_\ast = K_\ast^\top \alpha$ and the variance as
$\sigma_f^2 - \sum_i V_{ij}^2$ where $V = L^{-1} K_\ast$ (the Cholesky route to
$K_\ast^\top K_y^{-1} K_\ast$; see §3), with `include_noise` adding $\sigma_n^2$ for
observation-space prediction. `GaussianProcess.sample_posterior` draws paths via
$f = \mu_\ast + L_c z$ with $L_c$ the Cholesky factor of $\Sigma_\ast$ and
$z \sim \mathcal{N}(0, I)$.

---

## 3. Why Cholesky

The posterior equations contain $(K + \sigma_n^2 I)^{-1}$ three times. The library
never forms that inverse. Since $K_y = K + \sigma_n^2 I$ is symmetric positive
definite (a PSD Gram matrix plus a strictly positive diagonal shift), it factors as

$$
K_y = L L^\top,
$$

with $L$ lower triangular with positive diagonal — the Cholesky factorization,
$\approx n^3/3$ flops.

**Solve, don't invert.** Every appearance of $K_y^{-1}$ is a linear solve against the
factors:

$$
\alpha = K_y^{-1} y = L^{-\top} L^{-1} y
\quad\text{via}\quad
L u = y \ \text{(forward substitution)},\ \  L^\top \alpha = u \ \text{(back substitution)},
$$

each an $O(n^2)$ triangular solve. For the predictive variance, let
$V = L^{-1} K_\ast$ (one triangular solve with $m$ right-hand sides); then

$$
K_\ast^\top K_y^{-1} K_\ast = K_\ast^\top L^{-\top} L^{-1} K_\ast = V^\top V,
$$

whose diagonal is just the column-wise sum of squares of $V$. Two bonuses fall out:
each predictive variance is $\sigma_f^2$ *minus a sum of squares*, so in exact
arithmetic it can never exceed the prior variance (floating-point cancellation can
push it a hair below zero, hence the `1e-12` clamp in `predict`); and the quadratic
form in the marginal likelihood is $y^\top K_y^{-1} y = y^\top \alpha$, one dot
product.

**Conditioning: why the explicit inverse is worse.** With unit roundoff
$\varepsilon \approx 10^{-16}$ and condition number $\kappa = \kappa(K_y)$, solving
through a backward-stable factorization gives a relative error on the order of
$\kappa\  \varepsilon$. Forming $K_y^{-1}$ explicitly and then multiplying costs
roughly $3\times$ the flops, loses backward stability, and its worst-case error bound
scales like $\kappa^2 \varepsilon$ — inversion effectively squares the condition
number's bite. RBF Gram matrices are routinely terribly conditioned: in this repo's
own test data (12 points on $[0,10]$, $\ell = 1.5$, $\sigma_f^2 = 2$), the smallest
eigenvalue of $K$ is $\lambda_{\min} \approx 2 \times 10^{-9}$ while
$\lambda_{\max}$ is order $10$, so $\kappa \sim 10^{10}$. Then
$\kappa \varepsilon \sim 10^{-6}$ (six good digits survive a solve) but
$\kappa^2 \varepsilon \sim 10^{4}$ (the explicit-inverse worst case has no digits at
all). Same math, opposite outcomes.

**The determinant for free.** The marginal likelihood (§4) needs $\log\lvert K_y \rvert$.
From the factorization, $\lvert K_y \rvert = \lvert L \rvert \lvert L^\top \rvert = \left(\prod_i L_{ii}\right)^2$, so

$$
\log \lvert K_y \rvert = 2 \sum_{i=1}^n \log L_{ii}
$$

— no extra computation beyond reading the diagonal of a factor you already have.

**Jitter, and why it is principled.** When $\sigma_n^2$ is tiny and inputs are close
relative to $\ell$, rows of $K$ become nearly linearly dependent, $\lambda_{\min}$
sinks toward (or below) floating-point resolution, and Cholesky fails on a
nonpositive pivot. The fix is to add a small diagonal "jitter" $\delta I$ before
factorizing. This is not a hack on the linear algebra — it is a microscopic,
fully-interpretable **model change**:

$$
K + \sigma_n^2 I + \delta I = K + (\sigma_n^2 + \delta) I,
$$

i.e. exactly the same GP with noise variance $\sigma_n^2 + \delta$. Jitter means
"pretend the observations are a hair noisier than claimed." The code tries an
escalating ladder $\delta \in \lbrace 10^{-10}, 10^{-9}, \dots, 10^{-6} \rbrace$, warns on each
escalation, and raises only if $10^{-6}$ still fails — so the perturbation stays
minimal in the common case.

*When jitter is harmless — a concrete story from this repo's test suite.* Jitter is
invisible only when $\delta \ll \lambda_{\min}(K_y)$: the component of the fitted
training values along each eigendirection $v_i$ (eigenvalue $\lambda_i$) shrinks by
the relative factor $\delta / (\lambda_i + \delta)$, which is negligible only when
$\delta$ is tiny compared to the *smallest* eigenvalue. On the 12-point $[0,10]$ test data:

- At $\ell = 1.5$: $\lambda_{\min}(K) \approx 2 \times 10^{-9}$. The near-noiseless
  interpolation setting uses $\sigma_n^2 = 10^{-10}$, and the jitter floor adds
  another $10^{-10}$ — a total diagonal shift of about 10% of $\lambda_{\min}$.
  That visibly perturbs interpolation: the posterior mean misses the training targets
  by $\sim 10^{-3}$, purely for numerical (not correctness) reasons.
- At $\ell = 1.0$: $\lambda_{\min}(K) \approx 10^{-6}$, so the same
  $\sim 2 \times 10^{-10}$ shift is four orders of magnitude below $\lambda_{\min}$
  and completely invisible at the test's $10^{-4}$ tolerance.

This is exactly why the noiseless-interpolation test in `tests/test_gp.py` uses
$\ell = 1.0$ rather than the $\ell = 1.5$ used elsewhere — see the comment in that
test and the message of commit `4aac328`. Longer length scales make points look more
alike to the kernel, which drives $\lambda_{\min}$ down and makes the same jitter
relatively larger. "Jitter = a tiny extra noise term, harmless only when
$\lambda_{\min} \gg \delta$" is not a slogan; on this data you can watch it stop
being harmless.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess._update_factorization` — builds
$K_y$, runs the jitter ladder to get $L$, and caches $\alpha$ via `cho_solve`;
`predict` reuses $L$ through `solve_triangular`, and `log_marginal_likelihood` reads
$\log\lvert K_y \rvert$ off $\mathrm{diag}(L)$.

---

## 4. The log marginal likelihood

**Derivation.** The prior slice at the training inputs is $f \sim \mathcal{N}(0, K)$,
and $y = f + \varepsilon$ with independent $\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$.
Marginalizing out $f$ requires no integral gymnastics: a sum of independent Gaussian
vectors is Gaussian, with means and covariances adding, so

$$
y \mid X, \theta \ \sim\  \mathcal{N}(0,\  K_y), \qquad K_y = K + \sigma_n^2 I.
$$

This is the "marginal" likelihood because the latent function has been integrated out
— it is the probability the model as a whole (kernel, hyperparameters, noise,
averaged over every function the prior allows) assigns to the data actually seen.
Taking the log of the multivariate Gaussian density:

$$
\log p(y \mid X, \theta)
= \underbrace{-\tfrac{1}{2}\  y^\top K_y^{-1} y}_{\text{data fit}}
\ \underbrace{-\ \tfrac{1}{2} \log \lvert K_y \rvert}_{\text{complexity penalty}}
\ \underbrace{-\ \tfrac{n}{2} \log 2\pi}_{\text{constant}}.
$$

- **Data fit** is the (negative half) Mahalanobis norm of $y$ under the model — the
  only term that sees the actual target values. It is large (close to 0) when the
  covariance structure makes the observed $y$ look probable.
- **Complexity penalty**: $\lvert K_y \rvert$ is the volume of the prior's uncertainty
  ellipsoid over datasets. A flexible model (short $\ell$, large $\sigma_f^2$) can
  generate many different datasets, so its ellipsoid is fat and it pays
  $-\tfrac12 \log\lvert K_y \rvert$ for that flexibility.
- **Constant**: the $(2\pi)^{n/2}$ normalizer; irrelevant to optimization over $\theta$.

**The automatic Occam's razor.** $p(y \mid X, \theta)$ is a normalized density over
all datasets $y \in \mathbb{R}^n$ — it must integrate to 1. A model that can explain
everything spreads its unit of probability mass thinly; a model that concentrates
mass near datasets like the observed one scores higher *at* the observed one.
Maximizing the LML therefore trades fit against flexibility with no explicit
regularizer. Watch both failure modes on the length scale:

- **$\ell$ too small.** Off-diagonal kernel entries die, $K_y \to (\sigma_f^2 + \sigma_n^2) I$:
  the model claims $n$ *independent* Gaussian values. The fit term plateaus at
  $-\tfrac12 \lVert y \rVert^2 / (\sigma_f^2 + \sigma_n^2)$ — it extracts zero benefit
  from the data's smoothness. Meanwhile the penalty is maximal: by Hadamard's
  inequality, among PSD matrices with a given diagonal the determinant is largest for
  the diagonal matrix, so the independence model pays the biggest possible
  $\log\lvert K_y \rvert$ at that signal level. Smooth data explained as white noise
  = low density.
- **$\ell$ too large.** $K$ tends toward the rank-one matrix $\sigma_f^2 \mathbf{1}\mathbf{1}^\top$:
  spectrum $\approx \lbrace n\sigma_f^2 + \sigma_n^2, \sigma_n^2, \dots, \sigma_n^2 \rbrace$. The
  determinant collapses (great penalty term!), but everything in $y$ orthogonal to the
  constant vector — all the actual shape — must be explained through eigenvalues
  $\approx \sigma_n^2$, so the fit term costs
  $\approx -\tfrac12 \lVert y_\perp \rVert^2 / \sigma_n^2$, which is catastrophic for
  small noise. Inflating $\sigma_n^2$ to compensate turns the model into "constant
  plus big noise," which again wastes density on jagged datasets that did not occur.
- The maximum sits in between, where the correlation structure captures the real
  smoothness: residuals are small *and* the eigenvalues decay fast enough to keep the
  volume small. That interior optimum is the Occam trade made automatically.

**The analytic gradient (noted, not implemented).** With
$\alpha = K_y^{-1} y$, two matrix-calculus identities —
$\partial \log\lvert K_y \rvert = \mathrm{tr}(K_y^{-1}\  \partial K_y)$ and
$\partial K_y^{-1} = -K_y^{-1} (\partial K_y) K_y^{-1}$ — give

$$
\frac{\partial}{\partial \theta_j}\left(-\tfrac12 y^\top K_y^{-1} y\right)
= \tfrac12\  \alpha^\top \frac{\partial K_y}{\partial \theta_j} \alpha
= \tfrac12 \mathrm{tr}\left(\alpha \alpha^\top \frac{\partial K_y}{\partial \theta_j}\right),
$$

and combining with the determinant term:

$$
\frac{\partial \log p(y \mid X, \theta)}{\partial \theta_j}
= \tfrac12 \mathrm{tr}\left( \left(\alpha\alpha^\top - K_y^{-1}\right) \frac{\partial K_y}{\partial \theta_j} \right).
$$

This library optimizes the LML with finite-difference gradients inside L-BFGS-B
instead; wiring in this analytic gradient is the natural first extension (it reuses
the same Cholesky factors).

**How the code optimizes it.** `fit_hyperparameters` maximizes the LML over
$\log \theta$ — log space keeps all three parameters positive and puts their scales
on comparable footing — with multi-start L-BFGS-B inside spec bounds
(`DEFAULT_HP_BOUNDS`: $\ell \in [10^{-2}, 10]$, $\sigma_f^2 \in [10^{-2}, 10^2]$,
$\sigma_n^2 \in [10^{-8}, 1]$, stated for normalized inputs and standardized $y$).
The LML surface is multimodal (§7), hence the random restarts; the incoming $\theta$
is always kept as a candidate, so the fitted LML can never be worse than the starting
one; a $\theta$ whose $K_y$ defeats even the jitter ladder gets a large finite
penalty so L-BFGS-B just walks away from it.

**→ Code:** `src/gpbo/gp.py`, `GaussianProcess.log_marginal_likelihood` — evaluates
the three terms as $-\tfrac12 y^\top \alpha - \sum_i \log L_{ii} - \tfrac n2 \log 2\pi$
using the cached factorization; `GaussianProcess.fit_hyperparameters` — multi-start
L-BFGS-B on the negative LML in $\log\theta$.

---

## 5. Expected Improvement, derived

**Setup.** We maximize (convention fixed in §6). The incumbent is
$y_{\text{best}}$, the best observed value so far. At a candidate $x$, the GP
posterior for the objective is $f \sim \mathcal{N}(\mu, \sigma^2)$. Define the
improvement over the incumbent, demanding a margin $\xi \ge 0$:

$$
\mathrm{EI}(x) = \mathbb{E}\big[\max(f - y_{\text{best}} - \xi,\  0)\big].
$$

The $\max(\cdot, 0)$ is what makes this interesting: outcomes below the bar cost
nothing (we would simply keep the incumbent), so only the upside tail counts.

**Working the integral.** Assume $\sigma > 0$. Write $f = \mu + \sigma \epsilon$ with
$\epsilon \sim \mathcal{N}(0,1)$, and abbreviate the expected headroom and its
standardized version:

$$
I = \mu - y_{\text{best}} - \xi, \qquad z = \frac{I}{\sigma}.
$$

The integrand $\max(I + \sigma\epsilon, 0)$ is nonzero exactly when
$\epsilon > -I/\sigma = -z$, so

$$
\mathrm{EI}
= \int_{-z}^{\infty} (I + \sigma \epsilon)\  \phi(\epsilon)\  d\epsilon
= I \int_{-z}^{\infty} \phi(\epsilon)\  d\epsilon
\ +\  \sigma \int_{-z}^{\infty} \epsilon\  \phi(\epsilon)\  d\epsilon.
$$

First integral: $\int_{-z}^{\infty} \phi = 1 - \Phi(-z) = \Phi(z)$ by symmetry.
Second integral: since $\phi'(\epsilon) = -\epsilon\  \phi(\epsilon)$, the integrand
is an exact derivative, $\int_{-z}^{\infty} \epsilon\  \phi(\epsilon)\  d\epsilon
= \big[-\phi(\epsilon)\big]_{-z}^{\infty} = \phi(-z) = \phi(z)$. Therefore

$$
\boxed{\ \mathrm{EI}(x) = I\  \Phi(z) + \sigma\  \phi(z), \qquad
I = \mu - y_{\text{best}} - \xi,\quad z = I/\sigma. \ }
$$

**Reading the two terms.**

- $I\  \Phi(z)$ is **exploitation**: the mean's headroom over the bar, weighted by the
  probability $\Phi(z)$ that the point actually clears it. It dominates where the
  model already predicts something good.
- $\sigma\  \phi(z)$ is **exploration**: a reward purely for uncertainty. Even where
  $I < 0$ (mean below the incumbent), a large $\sigma$ keeps EI positive because the
  upper tail might reach past $y_{\text{best}}$. Whenever $\sigma > 0$, EI $> 0$.
- The split is exact, not a metaphor: differentiating the closed form gives
  $\partial \mathrm{EI} / \partial \mu = \Phi(z)$ (the $z$-dependent terms cancel:
  $I\phi(z)/\sigma + \sigma\phi'(z)/\sigma = z\phi(z) - z\phi(z) = 0$) and
  $\partial \mathrm{EI} / \partial \sigma = \phi(z)$ (again after exact cancellation).
  The marginal value of a better mean is a probability; the marginal value of more
  uncertainty is a density. Both are positive, so EI always wants both.
- $\xi$ raises the bar: larger $\xi$ discounts small mean advantages and tilts the
  balance toward exploration. The default $\xi = 0.01$ is in standardized-$y$ units
  (§6), i.e. one percent of a standard deviation of the observed values.

**The $\sigma \to 0$ limit — and where the code deliberately differs.** Take the
closed form's true limit at fixed $\mu$. If $I > 0$: $z \to +\infty$, so
$\Phi(z) \to 1$ and $\sigma\phi(z) \to 0$, giving $\mathrm{EI} \to I$. If $I < 0$:
$z \to -\infty$ and both terms vanish (the Gaussian tail decays faster than
$1/\sigma$ grows). If $I = 0$: $\mathrm{EI} = \sigma\phi(0) \to 0$. So

$$
\lim_{\sigma \to 0} \mathrm{EI} = \max(\mu - y_{\text{best}} - \xi,\  0):
$$

a point mass either beats the incumbent or it does not — no uncertainty, no
exploration credit.

The implementation does **not** return this limit. `expected_improvement` returns
$\mathrm{EI} = 0$ whenever $\sigma \le 10^{-12}$, *regardless of the mean*. That is a
deliberate Bayesian-optimization convention, not a bug: inside the BO loop, the only
places with (numerically) zero posterior uncertainty are already-observed points, and
the one thing the proposal mechanism must never do is re-propose a location it has
already evaluated — there is nothing left to learn there, and an acquisition argmax
stuck on the incumbent stalls the loop forever. Zeroing EI at zero-uncertainty
points enforces that; it also sidesteps computing $z = I/\sigma$ where it is
numerically undefined. If you compare the code to the math above and notice the
mismatch at $\sigma = 0$: it is intentional, and this paragraph is the reconciliation.
(The final `np.clip(ei, 0, None)` is separate and purely numerical: the exact
expression is provably nonnegative, but for very negative $z$ it is a difference of
two tiny floats and cancellation can leave $-10^{-18}$-type residue.)

**The noisy-objective caveat.** $y_{\text{best}}$ is the best *noisy* observation,
not the best latent value $f$. The maximum of $n$ noisy measurements is biased upward
(a selection effect: the winner is disproportionately likely to have gotten lucky
noise), so the incumbent is slightly optimistic and the bar EI compares against sits
a bit too high. With the small noise levels typical here the effect is minor;
noise-robust variants exist — e.g. replacing $y_{\text{best}}$ with the best
*posterior mean* over observed inputs, or acquisition functions designed for noise
(knowledge gradient) — and are out of scope for this library.

**→ Code:** `src/gpbo/acquisition.py`, `expected_improvement` — vectorized evaluation
of $I\ \Phi(z) + \sigma\ \phi(z)$ with the $\sigma \le 10^{-12} \Rightarrow \mathrm{EI} = 0$
convention and the final nonnegativity clip.

---

## 6. The BO loop and its conventions

The loop itself is short: fit the GP to everything seen, refit hyperparameters,
maximize EI over the box, apply the duplicate guard, evaluate the objective at the
proposal, append, repeat. Every design choice around it exists to keep the math of
§1–§5 honest.

**Why maximize-only.** One convention, one code path. Minimization is the caller's
one-liner: minimize $g$ by maximizing $-g$. Supporting both directions internally
would thread a sign through the incumbent ($\max$ vs $\min$), the improvement
definition, the EI formula, and `best_so_far` — quadrupling the surface for sign
bugs while adding zero capability.

**Why inputs are normalized to the unit box.** The optimizer maps the user's box
`bounds` to $[0,1]^d$ and runs everything there. The RBF kernel here is *isotropic* —
one shared length scale $\ell$ for all dimensions, applied to plain Euclidean
distance. If one coordinate ranged over $[0, 1000]$ and another over $[0, 0.1]$, no
single $\ell$ could be meaningful for both: distance would be entirely dominated by
the first coordinate. On the unit box, one length scale is a statement about
fractions of each search range ("correlations decay over about a third of the box"),
which is exactly the kind of prior a shared $\ell$ can express. It also makes the
fixed hyperparameter bounds ($\ell \in [10^{-2}, 10]$) and the warm-start value
$\ell = 0.3$ meaningful constants rather than data-dependent guesses.

**Why $y$ is standardized.** Before each fit, $y$ is rescaled to zero mean and unit
variance (recomputed every iteration, since new data shifts both). The GP prior has
mean zero and marginal variance $\sigma_f^2$; if the raw objective lived near, say,
$1000$, the zero-mean prior would be dishonest — far from data the posterior mean
reverts to $0$ (§2), a wild extrapolation that EI would read as either a catastrophic
or a fantastic region depending on sign. Standardization makes "revert to the prior"
mean "revert to the average of what we've seen," keeps $\sigma_f^2 \approx 1$ inside
its bounds, and gives $\xi = 0.01$ its interpretation as 1% of a standard deviation.
A guard substitutes $1$ for the standard deviation when it is numerically zero (e.g.
all initial observations equal).

**Maximizing EI.** In 1D the argmax is taken over a dense 1000-point grid — cheap,
deterministic, and exactly what the visualizations plot, so the plotted EI curve and
the proposals cannot disagree. For $d \ge 2$ a grid is exponentially hopeless, so the
code scores 2048 seeded random candidates and refines the top few with L-BFGS-B
inside the box.

**The duplicate guard.** By the §5 convention EI is zero at evaluated points, but
proposals come from the same fixed grid every iteration, so late in a converged 1D
run — when EI is essentially zero everywhere — the grid argmax repeatedly lands on
(or within numerical distance of) an already-evaluated node, typically the
incumbent. If the proposal is within $10^{-6}$ of an existing point, the guard
replaces it with a uniformly random point. Two consequences worth rehearsing: the
loop cannot stall re-evaluating one location, and it gets free extra exploration
precisely when the model claims there is nothing left to learn. So late proposals
that look random-uniform are expected behavior, not a bug. Bookkeeping honesty:
`IterationRecord.ei_max` records the EI of the *pre-guard* argmax, so on iterations
where the guard fired, `ei_max` describes the replaced proposal, not the random
`x_next` actually evaluated.

**Warm starts.** The same GP object persists across iterations, and
`fit_hyperparameters` always keeps the current $\theta$ as a candidate (§4), so each
iteration's hyperparameter search starts from last iteration's answer and can only
match or improve it under the new data.

**→ Code:** `src/gpbo/optimizer.py`, `BayesianOptimizer.run` — owns the loop and all
unit conversions (`_to_orig`, standardization); `_maximize_ei_grid` and
`_maximize_ei_candidates` — the two acquisition maximizers; `_apply_duplicate_guard`
— the $10^{-6}$ proximity fallback; `IterationRecord` — the per-iteration snapshot
consumed by the visualizations.

---

## 7. Limitations

**$O(n^3)$ scaling.** Every factorization costs $\approx n^3/3$ flops, and
hyperparameter fitting multiplies that: each LML evaluation inside L-BFGS-B is a full
refactorization, times finite-difference gradient evaluations, times restarts —
dozens to hundreds of Cholesky calls per BO iteration, and the loop refits every
iteration. The practical ceiling for exact GPs is a few thousand points on a laptop;
this library targets the regime BO actually lives in (expensive objectives, tens to a
few hundred evaluations), where $n^3$ is negligible. Beyond that ceiling the standard
escape routes are sparse/inducing-point approximations and iterative solvers — out of
scope here.

**Curse of dimensionality for BO.** Two independent problems in high $d$. First,
distance concentration: as $d$ grows, pairwise distances between random points
concentrate around a common value, so an isotropic RBF sees every point as roughly
equally far from every other — $K$ flattens, the posterior barely departs from the
prior anywhere, and EI goes flat, degrading BO toward random search. Second,
coverage: 2048 candidates are a dense sample of $[0,1]^2$ and a vanishingly sparse
one of $[0,1]^{20}$, so even a genuinely informative EI surface would be poorly
maximized. Rule of thumb: vanilla BO is credible up to roughly 10–20 dimensions;
past that you need structure (ARD length scales, additive models, trust regions,
random embeddings).

**LML multimodality.** The LML surface over $\theta$ routinely has several local
optima that are *genuinely different explanations of the data* — e.g. short-$\ell$,
low-noise "the function wiggles and we interpolate it" versus long-$\ell$,
high-noise "the function is a smooth trend and the wiggle is noise." Multi-start
L-BFGS-B (with the warm start always kept as a candidate) mitigates the risk of
landing in a bad mode but guarantees nothing; the fitted $\theta$ can also jump
between explanations across BO iterations as new data tips the balance. The fully
principled treatment — posterior inference over $\theta$ (e.g. MCMC) instead of a
point estimate — is out of scope.

**→ Code:** the ceilings live where the costs live — `src/gpbo/gp.py`
`_update_factorization` (the $n^3/3$ inner loop), `src/gpbo/optimizer.py`
`_maximize_ei_candidates` (the coverage problem in $d \ge 2$), and `src/gpbo/gp.py`
`fit_hyperparameters` (the multi-start answer to multimodality).
