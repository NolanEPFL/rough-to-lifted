# Model specification: Lifted Heston vs aBergomi on SPX

> This document is the mathematical specification behind the code: the equations,
> numerical schemes, hyperparameters, and sanity checks that `src/` implements. It
> is the reference companion to the thesis
> [`From_Rough_to_Lifted.pdf`](From_Rough_to_Lifted.pdf) and to the annotated
> bibliography [`../PAPERS.md`](../PAPERS.md).

---

## 0. Project goal

Compare two finite-dimensional Markovian approximations of rough volatility models on
SPX implied volatility surfaces:

1. **Lifted Heston** (Abi Jaber, *Lifting the Heston model*, 2019, arXiv 1810.04868) —
   square-root variance, geometric Lévy-measure kernel, Riccati + Fourier (COS) pricing.
2. **aBergomi** (Zhu, Loeper, Chen, Langrené, 2021) — lognormal variance, L²-optimised
   exponential-sum kernel, Monte Carlo pricing.

The comparison is performed in the **same forward-variance specification** so the two
models share the same input curve `ξ₀(·)` and the same effective parameter dimension.

This is the empirical contribution of the bachelor thesis: no published paper has done
this comparison systematically. Abi Jaber & Li (2025) compare rough/path-dependent/
Markovian *within* the Bergomi family but never against any Heston-family model.

---

## 1. Notation, conventions, and global assumptions

- **Measure.** All pricing is under the risk-neutral measure Q. We work directly in the
  forward measure: discount factor and dividends are zero (`r = q = 0`). Spot is `S₀`,
  log-spot `X_t = log S_t`, log-moneyness `k = log(K/F_T)` with `F_T = S₀` here.
- **Hurst index.** `H ∈ (0, 1/2)`. We additionally use `α := H + 1/2 ∈ (1/2, 1)`.
- **Rough kernel.**
  $$ K_H(t) = \frac{t^{H-1/2}}{\Gamma(H+1/2)} = \frac{t^{\alpha-1}}{\Gamma(\alpha)}. $$
  `K_H` is singular at `t = 0` for `H < 1/2`; any kernel-fitting integration must use a
  positive lower bound `ε > 0`.
- **Sum-of-exponentials approximation.**
  $$ K_n(t) = \sum_{i=1}^n c_i^n \, e^{-x_i^n t}, \qquad c_i^n > 0,\ x_i^n > 0. $$
  Throughout the code, we maintain `c_i, x_i > 0` strictly (never `≥ 0`); allowing zero
  speeds breaks several closed-form formulas (e.g., `(1−e^{−xT})/x → T`).
- **Forward variance curve.** `ξ₀(·): [0, T_max] → ℝ_+`. For synthetic experiments we
  use a flat curve `ξ₀(t) = V₀`. For SPX experiments we extract `ξ₀` from variance
  swap quotes; see Section 9.
- **Time unit.** Years. SPX maturities given in days are divided by 365.

### Library choices

- Python 3.11+, `numpy`, `scipy` (`integrate`, `optimize`, `stats`, `interpolate`),
  `pandas`, `matplotlib`. Optional: `numba` for the OU step inner loop, `pytest` for
  tests, `tqdm` for progress.
- **No JAX, no PyTorch.** The thesis is implementation-clarity-first; speed is a
  secondary concern.

---

## 2. The Lifted Heston model

### 2.1 SDE definition (eqs. 4.1–4.3 of the thesis; (2.1)–(2.3) of Abi Jaber 2019)

For fixed `n ∈ ℕ`, parameters `λ ≥ 0`, `ν > 0`, `ρ ∈ [-1, 1]`, weights `{c_i}_{i=1}^n > 0`,
mean-reversion speeds `0 < x_1 < … < x_n`, and admissible curve `g₀ⁿ`,

$$
\begin{aligned}
dS_t &= S_t \sqrt{V_t^n}\, dB_t, \\
V_t^n &= g_0^n(t) + \sum_{i=1}^n c_i^n U_t^{n,i}, \\
dU_t^{n,i} &= \bigl(-x_i^n U_t^{n,i} - \lambda V_t^n\bigr) dt + \nu \sqrt{V_t^n}\, dW_t,
\quad U_0^{n,i} = 0,
\end{aligned}
$$
with `B = ρ W + √(1−ρ²) W^⊥`.

### 2.2 Geometric/Lévy parametrisation (eqs. 4.74–4.75; (3.3) of Abi Jaber 2019)

For `n` even, `r_n > 1`, partition `(0, ∞)` with endpoints `η_i^n = r_n^{i − 1 − n/2}`,
`i = 0, …, n`. Letting `α := H + 1/2`,

$$
\begin{aligned}
c_i^n &= \frac{(r_n^{1-\alpha} - 1)\, r_n^{(\alpha-1)(1+n/2)}}{\Gamma(\alpha)\,\Gamma(2-\alpha)}\,
        r_n^{(1-\alpha)\, i}, \\[4pt]
x_i^n &= \frac{1-\alpha}{2-\alpha} \cdot \frac{r_n^{2-\alpha} - 1}{r_n^{1-\alpha} - 1}
        \cdot r_n^{i - 1 - n/2}.
\end{aligned}
$$

> **Sanity at `H = 1/2`.** Both formulas have removable singularities at `α = 1`. Code
> must handle this either by analytic limits or by clamping `H` away from `0.5 − ε`.
> For the thesis empirical work `H ∈ (0.02, 0.49)`, so this is a unit-test concern only.

**Default hyperparameters.** `n = 20`, `r_n = 2.5` (Abi Jaber's empirical sweet spot,
matches rough Heston IV surface to within ~10⁻⁵ MSE while ~20× faster).

### 2.3 Forward-variance specification (`λ = 0`)

When `λ = 0`, the Volterra equation for the expected variance (Prop. 4.2 of the thesis)
becomes `m_n(t) = g_0^n(t)`. Setting `g_0^n(t) := ξ_0(t)` yields **`E^Q[V_t^n] = ξ_0(t)`**
exactly. This is the version we use for all empirical comparisons.

In this specification, the Riccati function `F` simplifies (drop `−λv`):

$$
F(u, v) = \tfrac{1}{2}(u^2 - u) + \rho \nu u\, v + \tfrac{\nu^2}{2} v^2.
$$

The free parameters reduce to `(ν, ρ, H)` once `(n, r_n)` and `ξ₀` are fixed.

### 2.4 Riccati ODE system and characteristic function

For `u ∈ ℂ` with `Re(u) ∈ [0, 1]`, define `ψ^{n,i}: [0, T] → ℂ` by (eq. 4.35):

$$
\frac{d\psi^{n,i}}{dt}(t) = - x_i^n \psi^{n,i}(t) + F\!\left(u, \sum_{j=1}^n c_j^n \psi^{n,j}(t)\right),
\quad \psi^{n,i}(0) = 0.
$$

Then the characteristic function (eq. 4.34, with `λ = 0` and `g_0^n = ξ₀`) is

$$
\Phi_T(u) := \mathbb{E}^Q[e^{u \log S_T}] = \exp\!\Bigl(u \log S_0 + \phi^n(0, T)\Bigr),
$$

where (eq. 4.37 with `g_0^n = ξ₀`)

$$
\phi^n(0, T) = \int_0^T F\!\left(u, \sum_{j=1}^n c_j^n \psi^{n,j}(s)\right) \xi_0(T - s)\, ds.
$$

> **Initial values of factors are zero**, so the term `Σ c_i ψ^{n,i}(T) U_0^{n,i}` vanishes
> at `t = 0`. This is why `Φ_T(u)` has the simple form above.

### 2.5 Numerical integration of the Riccati system (Appendix C of Abi Jaber 2019)

The system is **stiff** because `x_n^n` grows large with `n`: e.g., `n = 20`, `r = 2.5`,
`H = 0.1` gives `x_{20}^{20} ≈ 6418`. The explicit Euler scheme has stability constraint
`Δt ≤ min_i (1/x_i^n)`, which is prohibitive.

Use the **explicit-implicit scheme** (Abi Jaber Appendix C, eq. C.2):

$$
\hat\psi_0^{n,i} = 0, \qquad
\hat\psi_{t_{k+1}}^{n,i} = \frac{1}{1 + x_i^n \Delta t}
\Bigl[ \hat\psi_{t_k}^{n,i} + \Delta t\, F\!\left(u, \sum_j c_j^n \hat\psi_{t_k}^{n,j}\right) \Bigr].
$$

This is unconditionally stable in `Δt`. For `T = 2` years use `N = 200` time steps as
default; verify convergence with `N ∈ {100, 200, 400, 800}`.

`φ^n(0, T)` is computed by trapezoidal quadrature of `F(u, Ψ_k) · ξ₀(T − t_k)` over the
same grid where `Ψ_k = Σ_j c_j ψ_k^{n,j}`.

### 2.6 Pricing via the COS method (Fang & Oosterlee 2009)

For a European call with strike `K` and maturity `T` on a forward `F = S₀` (zero rates),

$$
C(K, T) = e^{-rT} \cdot \sum_{j=0}^{N_{\text{cos}}-1} {}^\prime\, \mathrm{Re}\!\Bigl[ \Phi_T\!\bigl(\tfrac{j\pi}{b-a}\bigr)\, e^{-i j \pi a/(b-a)} \Bigr] \cdot V_j(K, T),
$$

where `Σ′` means the `j = 0` term has weight `1/2`, `[a, b]` is the truncation interval
for the log-asset density, `Φ_T` is the characteristic function of `log(S_T / S_0)` (i.e.,
remove the `u log S_0` term: pass `u = ij π/(b−a)`, evaluate `exp(φ^n(0, T))`), and
`V_j(K, T)` are the Fourier-cosine coefficients of the call payoff:

$$
V_j(K, T) = \frac{2 K}{b-a}\bigl( \chi_j(0, b) - \psi_j(0, b) \bigr),
$$

with the standard formulas (Fang & Oosterlee Section 2):

$$
\chi_j(c, d) = \frac{1}{1 + (j\pi/(b-a))^2} \Bigl[ \cos\!\bigl(\tfrac{j\pi(d-a)}{b-a}\bigr) e^d
- \cos\!\bigl(\tfrac{j\pi(c-a)}{b-a}\bigr) e^c
+ \tfrac{j\pi}{b-a} \sin\!\bigl(\tfrac{j\pi(d-a)}{b-a}\bigr) e^d
- \tfrac{j\pi}{b-a} \sin\!\bigl(\tfrac{j\pi(c-a)}{b-a}\bigr) e^c \Bigr],
$$

$$
\psi_j(c, d) = \begin{cases}
\frac{(b-a)}{j\pi}\bigl[ \sin\!\bigl(\tfrac{j\pi(d-a)}{b-a}\bigr) - \sin\!\bigl(\tfrac{j\pi(c-a)}{b-a}\bigr)\bigr] & j \neq 0, \\
d - c & j = 0.
\end{cases}
$$

**Truncation interval `[a, b]`.** For Heston-like models, use the rule of thumb
`[a, b] = [c_1 − L √c_2, c_1 + L √c_2]` with `L = 12` and where `c_1, c_2` are the first
two cumulants of `log(S_T/S_0)`. For lifted Heston we use the simpler conservative
choice
$$ [a, b] = [-L_0 \sqrt{T \cdot \xi_0(T)}, +L_0 \sqrt{T \cdot \xi_0(T)}], \quad L_0 = 12, $$
which is wide enough for typical SPX surfaces. **Always verify** by checking that the
COS price is stable when `L_0` is increased to `15`.

**Number of cosine terms.** `N_cos = 256` is sufficient for IV accuracy ~10⁻⁵; use
`512` if you observe Gibbs-style oscillations.

### 2.7 Black–Scholes implied volatility inversion

Use Brent's method on the Black–Scholes price minus the model price, with bracket
`[10⁻⁶, 5.0]`. Use `scipy.optimize.brentq`.

For numerical stability:
- Skip strikes where the call price violates intrinsic-value bounds (`max(F − K, 0) ≤ C ≤ F`).
- For deep OTM, use the put price via put-call parity then invert.

---

## 3. The aBergomi model

### 3.1 Definition (eqs. 5.3–5.5 of the thesis)

Replace the rough kernel by an exponential-sum approximation:

$$
K_H(t) \approx \tilde K_n(t) = \sum_{i=1}^n \tilde c_i\, e^{-\tilde x_i t}.
$$

The variance process is

$$
V_t = \xi_0(t) \exp\!\Bigl(\eta\, Y_t - \tfrac{\eta^2}{2}\, \mathrm{Var}(Y_t)\Bigr),
\qquad Y_t = \sum_{i=1}^n \tilde c_i\, X_t^{(i)},
$$

where each `X_t^{(i)}` is the OU-type Gaussian process

$$
X_t^{(i)} = \int_0^t e^{-\tilde x_i (t - s)}\, dW_s,
$$

driven by **the same** Brownian motion `W` for all `i`. The stock satisfies

$$
d\log S_t = -\tfrac{1}{2} V_t\, dt + \sqrt{V_t}\, dB_t,
\quad B_t = \rho W_t + \sqrt{1 - \rho^2}\, W_t^\perp.
$$

The free parameters are `(η, ρ, H)`, with `(n, ε, T_max)` and the kernel-fitting weight
`w(t)` as fixed hyperparameters. By construction `E^Q[V_t] = ξ_0(t)`.

### 3.2 Variance of `Y_t` (closed form)

Since `W` drives all factors and they are deterministic OU integrals,

$$
\mathrm{Cov}(X_t^{(i)}, X_t^{(j)}) = \int_0^t e^{-(\tilde x_i + \tilde x_j)(t-s)}\, ds
= \frac{1 - e^{-(\tilde x_i + \tilde x_j) t}}{\tilde x_i + \tilde x_j},
$$

hence

$$
\mathrm{Var}(Y_t) = \sum_{i, j} \tilde c_i \tilde c_j \frac{1 - e^{-(\tilde x_i + \tilde x_j) t}}{\tilde x_i + \tilde x_j}.
$$

This is computed once per maturity grid and cached.

### 3.3 L² kernel fit (eq. 5.4)

Solve

$$
(\tilde c_i, \tilde x_i)_{i=1}^n = \arg\min_{c_i, x_i > 0} \int_\varepsilon^{T_{\max}}
\Bigl( K_H(t) - \sum_{i=1}^n c_i e^{-x_i t} \Bigr)^2 w(t)\, dt.
$$

**Defaults.** `ε = 1/365` (one day), `T_max = 2.0` years (longer than the longest SPX
maturity of interest), `w(t) ≡ 1`.

**Stable parametrisation.**
1. Optimise log-speeds: `x_i = exp(y_i)` (no positivity constraint needed on `y`).
2. For fixed `{x_i}`, weights solved by **non-negative least squares (NNLS)** on the
   discretised integral (use `scipy.optimize.nnls` on a fine quadrature grid of
   100–200 points log-spaced in `[ε, T_max]`).
3. Outer optimisation on `{y_i}` via `scipy.optimize.minimize` (L-BFGS-B). Use
   **5 random restarts** with log-speeds drawn uniformly in `[log(0.1), log(1000)]`,
   take the best. Sort the result by `x_i` ascending.

This is more stable than optimising `2n` variables jointly.

### 3.4 Geometric kernel for aBergomi (kernel-controlled comparison)

For Experiment 3 (kernel-controlled), aBergomi uses the **same** geometric grid as
lifted Heston (Section 2.2). This is mathematically legitimate: the aBergomi structure
requires only positive `(c_i, x_i)` and any positive sum-of-exponentials kernel works.
The "canonical aBergomi" interpretation is then identical to that of lifted Heston —
both indexed by `H` only.

### 3.5 Exact Monte Carlo simulation of the OU factors

Time grid `0 = t_0 < t_1 < … < t_N = T` with constant `Δ`. The exact one-step recursion is

$$
X_{t+\Delta}^{(i)} = e^{-\tilde x_i \Delta} X_t^{(i)} + G_t^{(i)},
$$

where the increment vector `G_t = (G_t^{(1)}, …, G_t^{(n)})` is **jointly Gaussian** with

$$
\mathrm{Cov}(G_t^{(i)}, G_t^{(j)}) = \int_t^{t+\Delta} e^{-(\tilde x_i + \tilde x_j)(t+\Delta - s)} ds
= \frac{1 - e^{-(\tilde x_i + \tilde x_j) \Delta}}{\tilde x_i + \tilde x_j}.
$$

This `n × n` matrix `Σ_step` is **time-independent** (depends only on `Δ`), so compute its
Cholesky factor `L_step` once per simulation. Sampling `G = L_step · Z` for `Z ~ N(0, I_n)`.

### 3.6 Joint simulation of `(W, B)` and the stock

The Brownian motion driving the variance is `W`. The Brownian motion driving the stock
is `B = ρ W + √(1−ρ²) W^⊥`. We need the **same** path of `W` to appear in both the
variance OU update and the stock log-return.

Let `ΔW_t` denote the increment of `W` over `[t, t+Δ]`. The OU increments `G_t` are
**not independent** of `ΔW_t`; in fact

$$
G_t^{(i)} = \int_t^{t+\Delta} e^{-\tilde x_i (t+\Delta - s)}\, dW_s.
$$

Two acceptable simulation strategies:

**Strategy A (recommended).** Simulate the vector
`(ΔW_t, G_t^{(1)}, …, G_t^{(n)})` jointly Gaussian with covariances
- `Var(ΔW) = Δ`,
- `Cov(ΔW, G^{(i)}) = ∫_t^{t+Δ} e^{-x_i (t+Δ - s)} ds = (1 − e^{−x_i Δ}) / x_i`,
- `Cov(G^{(i)}, G^{(j)})` as above.

This is an `(n+1) × (n+1)` covariance matrix, time-independent, Cholesky once.

**Strategy B.** Simulate `ΔW_t ~ N(0, Δ)`, then condition `G_t | ΔW_t`. The conditional
distribution is Gaussian; mean is a known affine function of `ΔW_t`, covariance is
the Schur complement. Equivalent to A, just less elegant.

**Use Strategy A.**

The stock update over one step:

$$
\log S_{t+\Delta} = \log S_t - \tfrac{1}{2} V_t \Delta + \sqrt{V_t}\, \bigl(\rho \Delta W_t + \sqrt{1-\rho^2}\, \Delta W_t^\perp\bigr),
$$

where `ΔW^⊥ ~ N(0, Δ)` is **independent** of all the `G_t` and of `ΔW_t`. (Two passes of
Gaussian sampling per step, but that's fine.)

> **Important.** `V_t` here is computed from `X^{(1)}_t, …, X^{(n)}_t` via the formula in
> Section 3.1, evaluated at the *current* time `t` and using the variance correction
> `Var(Y_t)` at time `t`. Pre-compute `Var(Y_t)` for every `t_k` on the time grid.

### 3.7 Pricing

For a single maturity `T`:

1. Simulate `M` paths of `(X^{(1)}, …, X^{(n)}, log S)` up to time `T` (ideally with
   antithetic variates: pair each `Z` with `−Z`, so effective paths are `2M`).
2. For each strike `K` in the grid, compute the call payoff `(S_T − K)^+` and average.
3. Standard error from the empirical variance / √M.

For a maturity grid `T_1 < T_2 < … < T_M`: simulate one path long enough to reach
`T_M`, record `S_{T_m}` for each `m`. Memory cost is `O(M_paths × M)`, time cost is
`O(M_paths × N_steps × n_factors)`.

**Defaults.**
- `M_paths = 50_000` for development, `200_000` for final results.
- `n_steps = 100 × T_max` (i.e., one step every ~3.6 days for `T_max = 2`).
- Antithetic variates: yes.
- **Sobol QMC**: yes for production runs (use `scipy.stats.qmc.Sobol`, generate
  `2 × M_paths × N_steps × (n+1)` standard normals via Box-Muller from Sobol points;
  scramble with `seed` for replication).
- **Common random numbers**: yes during calibration. Generate `Z` once, store in
  memory or memmap, reuse across all loss evaluations.

---

## 4. The Bergomi–Guyon ATM skew benchmarks

These are first-order in vol-of-vol approximations. Use them as sanity checks against
the calibrated models, **not** as the calibration objective.

### 4.1 Lifted Heston (eq. 4.77 of the thesis)

$$
\psi_{\text{LH}}(T) \approx \frac{\rho \nu}{2 \sqrt{V_0}\, T} \sum_{i=1}^n c_i^n
\frac{1 - e^{-x_i^n T}}{x_i^n}.
$$

### 4.2 aBergomi (eq. 5.6)

$$
\psi_{\text{aB}}(T) \approx \frac{\rho \eta}{2 T} \sum_{i=1}^n \tilde c_i
\frac{1 - e^{-\tilde x_i T}}{\tilde x_i}.
$$

> **Note.** The two formulas differ structurally only by the `1/√V_0` factor in lifted
> Heston (because the leverage acts via `√V` in the SDE) and the choice of `(c_i, x_i)`.

### 4.3 Reference rough power-law

$$
\psi_{\text{rough}}(T) \approx C \cdot T^{H - 1/2}.
$$

For `T → 0`, all finite-`n` Markovian approximations have a **bounded** skew, while the
rough limit blows up. This is the structural difference that the synthetic experiments
should expose.

---

## 5. The rough Heston benchmark (optional, for convergence study only)

`Vt` solves the fractional Volterra SDE:

$$
V_t = V_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} \kappa(\theta - V_s) ds
+ \frac{\nu}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} \sqrt{V_s}\, dW_s.
$$

Pricing uses the El Euch–Rosenbaum fractional Riccati:

$$
g(u, t) = \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} F(u, g(u, s))\, ds,
$$

solved by the **Adams predictor–corrector** scheme of Diethelm–Ford. Use a fine grid
(`N = 1000` for `T = 2`) as the reference for convergence studies.

This is **only** used in `scripts/01_lh_convergence.py` to validate that lifted Heston
converges to rough Heston as `n → ∞`. It is **not** part of the empirical comparison
with aBergomi.

> **Time budget.** If implementing rough Heston is taking more than 2 days, *skip it*
> and replace the convergence reference with `n = 500` lifted Heston as a proxy. The
> thesis claim is that `n = 20` is close to `n → ∞`, not specifically to rough Heston.

---

## 6. Calibration

### 6.1 Loss function

Weighted RMSE on implied volatilities:

$$
\mathrm{Loss}(\theta) = \sqrt{\frac{\sum_{m, j} w_{m, j}\,
\bigl[\sigma^{\text{model}}(K_j, T_m; \theta) - \sigma^{\text{mkt}}(K_j, T_m)\bigr]^2}
{\sum_{m, j} w_{m, j}}}.
$$

Two weight schemes:
- **Equal weights.** `w_{m,j} = 1`. Default for the main report.
- **Vega weights.** `w_{m,j} = vega_BS(K_j, T_m, σ_mkt(K_j, T_m))`. Reported as a
  robustness check.

Add a **skew loss**: define `S^*(T) := ∂σ/∂k|_{k=0}` from a finite difference (use the
two strikes nearest ATM), and add `λ_skew * RMSE(skew)` with `λ_skew = 0.1`. Default
off; report on/off comparison.

### 6.2 Parameter bounds

Lifted Heston (forward-variance form): `(ν, ρ, H) ∈ [0.05, 3.0] × [-0.99, 0.0] × [0.02, 0.49]`.

aBergomi: `(η, ρ, H) ∈ [0.5, 5.0] × [-0.99, 0.0] × [0.02, 0.49]`.

### 6.3 Optimisation strategy

1. **Differential evolution** for global search: `popsize = 10`, `maxiter = 30`, fixed seed.
2. **L-BFGS-B refinement** starting from the DE optimum: `maxiter = 100`, finite-difference
   gradient with `eps = 1e-4`.
3. For aBergomi: **freeze CRN** (Sobol seed + `Z` matrix) at the start of calibration
   and reuse for every loss evaluation. Compute final reported MC standard errors with a
   **fresh, larger** independent sample (`M = 200_000`).

### 6.4 Out-of-sample validation

Calibrate on `T ≤ 1` year, report fit and skew metrics on `T > 1` year. This is a
modest split but it converts a descriptive table into an honest predictive comparison.

---

## 7. Experiments to run

### Experiment 1: Lifted Heston convergence study (`scripts/01_lh_convergence.py`)

For `H ∈ {0.05, 0.10, 0.20, 0.30}` and `n ∈ {1, 3, 5, 10, 20, 50}`:
- Compute IV surface for fixed `(ν, ρ) = (0.4, -0.7)`, flat `ξ₀ = 0.04`.
- Reference: `n = 500` lifted Heston (or rough Heston if implemented).
- Report `sup_{K,T} |σ_n - σ_ref|` and `RMSE` as a function of `n`.

### Experiment 2: Kernel approximation error in isolation (`scripts/02_kernel_study.py`)

For `H ∈ {0.05, 0.10, 0.20, 0.30}` and `n ∈ {3, 5, 10, 20}`:
- Compute geometric `K_n` and L²-fit `K_n`.
- Report `‖K_H − K_n‖_{L²([ε, T_max])}` and `sup`-norm.
- Tells us whether the kernel difference alone explains any model gap.

### Experiment 3: Synthetic comparison (`scripts/03_synthetic_comparison.py`)

Generate a "true" surface from rough Heston (or `n = 500` lifted Heston). Calibrate
both Lifted-Heston-`n=20` and aBergomi to it. Report fit RMSE and parameter recovery.
Sanity check that both can match a ground truth they're approximating.

### Experiment 4: SPX comparison (`scripts/04_spx_comparison.py`) — main result

For one well-cleaned SPX date:
- Three calibrations: LH-geo, aB-L², aB-geo.
- Report: surface RMSE (in-sample, out-of-sample), ATM skew RMSE, calibration time,
  pricing time per surface, calibrated `(ν or η, ρ, H)`.
- Plots: (a) market vs model IV at 5 maturities, (b) ATM skew term-structure for all
  models + market, (c) residuals heatmap on `(k, T)`.

> **Optional (do only if time permits) — Experiment 5: LH-L² robustness check.**
> Add LH with the L² kernel as a fourth column. Verifies whether the kernel choice
> (rather than the variance dynamics) explains the lifted-vs-aBergomi gap.

---

## 8. Sanity checks (must all pass before any empirical run)

### 8.1 Lifted Heston

- [ ] `Φ_T(0) = 1` exactly for any parameters and any `T > 0`.
- [ ] **Heston recovery.** `n = 1`, `c_1 = 1`, `x_1 = 0`, `λ > 0`, `g_0(t) = V_0 + λθ t`:
      compare characteristic function values at a grid of `u ∈ ℂ` against a textbook
      Heston implementation. Match to 10⁻⁶ relative.
- [ ] **Convergence in `n`.** For fixed `(H, ν, ρ)`, IV surface stable to within 10⁻⁴
      between `n = 100` and `n = 200`.
- [ ] **Put-call parity.** Compute `C(K, T)` and `P(K, T)` via COS, verify
      `C - P = S_0 - K e^{-rT}` to 10⁻⁸.
- [ ] **No-arbitrage.** Calls monotone decreasing in `K`, convex in `K`.

### 8.2 aBergomi

- [ ] `E^Q[V_t] ≈ ξ_0(t)` empirically: simulate `M = 100_000` paths, average `V_t` at a
      grid of `t`, compare to `ξ_0(t)`. Relative error < 1% for `t ∈ [0.05, 2]`.
- [ ] **MC standard error decreases as `1/√M`** on a log-log plot.
- [ ] **Calls monotone in `K`, convex in `K`** (no MC-induced arbitrage at reasonable `M`).
- [ ] **Black–Scholes recovery at `η = 0`**: should give BS prices with `σ² = ξ_0(T)` (use
      `T` for terminal Black–Scholes, not the integrated variance — at `η = 0`, `V_t = ξ₀(t)`,
      so total variance is `∫₀^T ξ₀(t) dt`).

### 8.3 Kernel module

- [ ] `K_H(t)` matches `t^{H-1/2} / Γ(H+1/2)` numerically for `t ∈ [10⁻³, 10]`,
      `H ∈ {0.05, 0.1, 0.2, 0.3}`.
- [ ] L² kernel fit reduces error monotonically in `n`.
- [ ] All fitted weights `c_i > 0`, all speeds `x_i > 0`.

### 8.4 COS pricing

- [ ] Test against a closed-form Heston call (Albrecher et al. 2007 reference).
- [ ] Stable when `L_0` increased from 12 to 15.
- [ ] Stable when `N_cos` increased from 256 to 512.

---

## 9. SPX data preparation (`src/data/`)

For the bachelor scope, **one well-cleaned SPX date** is sufficient.

### 9.1 Required raw inputs

A CSV with columns: `expiry_date`, `strike`, `bid`, `ask`, `option_type` (C/P), `quote_date`,
`spot`, `risk_free_rate` (or zero curve to interpolate).

### 9.2 Cleaning steps

1. **Mid-price.** `mid = (bid + ask) / 2`.
2. **Filter.** Keep options with `bid > 0`, `ask > 0`, `bid/ask spread < 50% of mid`,
   strikes in `[0.5, 2.0] × forward`.
3. **Forward extraction.** For each maturity, use put-call parity at the strike `K*`
   minimising `|C - P|`: `F = K* + e^{rT}(C - P)|_{K*}` (Brent's method on the
   call-put difference if needed).
4. **OTM only.** Use OTM puts (`K < F`) and OTM calls (`K > F`).
5. **Implied vol inversion.** Brent on Black–Scholes price.
6. **No-arbitrage filtering.** Drop strikes where call prices violate monotonicity in
   `K` after smoothing (a few outliers per slice are common).
7. **Forward variance curve `ξ₀`.** Either: (a) use variance swap quotes if available,
   or (b) fit an SVI (or SSVI) per slice, integrate the model variance, take the
   numerical derivative as in Bergomi & Guyon (and the 2025 Abi Jaber & Li paper).

### 9.3 Synthetic alternative

If real SPX data is not available, generate a synthetic surface from rough Heston with
known parameters and use it as the "market". This is a degraded but still valid
comparison (Experiments 1–3 only; Experiment 4 becomes an internal-consistency test).

---

## 10. Output and reporting

### 10.1 Tables (LaTeX)

Each script that produces a result must emit a `.tex` snippet in `results/tables/`.

### 10.2 Figures

PDF format, vector graphics, no titles (titles go in LaTeX captions). Use
`matplotlib.rcParams` to set: `font.family = 'serif'`, `font.size = 10`,
`figure.figsize = (5.5, 3.5)`. Save with `bbox_inches='tight'`.

### 10.3 Reproducibility

Every script accepts a `--seed` argument (default `42`) and saves to
`results/<experiment_name>/<git_hash>_<seed>/`. Save inputs (parameter dict, hyperparameter
dict) as JSON alongside outputs.

---

## 11. Precise references used in this project

- **[AJ2019]** Abi Jaber, E. (2019). *Lifting the Heston model.* Quantitative Finance.
  arXiv:1810.04868. Equations (2.1)–(2.3) for the SDE, (2.9)–(2.10) for the Riccati,
  (3.3) for the geometric grid, Appendix C for the explicit-implicit scheme.
- **[ZLCL2021]** Zhu, Loeper, Chen, Langrené (2021). *aBergomi: Markovian approximation*.
- **[AJL2025]** Abi Jaber & Li (2025). *Volatility models in practice: rough,
  path-dependent, or Markovian?* arXiv:2401.03345.
- **[CD2023]** Cont & Das (2023). *Rough volatility: fact or artefact?*
- **[FO2009]** Fang & Oosterlee (2009). *A novel pricing method for European options
  based on Fourier-cosine series expansions.* SIAM J. Sci. Comp.
- **[ER2018]** El Euch & Rosenbaum (2018). *The characteristic function of rough Heston.*
  Mathematical Finance.

---

*End of CONTEXT.md.*
