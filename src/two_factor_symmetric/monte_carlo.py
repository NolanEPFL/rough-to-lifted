"""Euler–Maruyama simulator for the symmetric-split two-factor lifted Heston.

Used only for the Monte Carlo sanity checks (the model is priced by COS). Both
blocks are now lifted-Heston OU-factor blocks (UNLIKE the asymmetric model, whose
block 2 is a CIR factor), each with a non-negative additive forcing:

    V^(j)_t = g0^(j)(t) + Σ_i c_i^(j) U^(j,i)_t,   g0^(1)=(1-w)ξ0, g0^(2)=w ξ0
    dU^(j,i)_t = -x_i^(j) U^(j,i)_t dt + nu_j √V^(j)_t dW_j,   U_0 = 0
    dS_t/S_t = √V^(1)_t dB^(1)_t + √V^(2)_t dB^(2)_t,
    B^(j) = rho_j W_j + √(1-rho_j²) W_j^perp,  (W1,W1perp,W2,W2perp) independent.

Numerical choices (matched to the existing two-factor MC validator):
  - Full truncation: √V uses max(V_raw, 0) where V_raw = g0^(j)+Σ c_i U_i.
  - Exact conditional OU step for each factor U^(j,i) (removes the large Euler
    bias for fast factors x_i·dt ≫ 1), with the noise integral decomposed into a
    part correlated with dW_j and an independent residual:
        U_{k+1} = U_k e^{-x dt} + nu_j √V^(j) (cov_c · dW_j + sigma_res · eps),
        cov_c = (1-e^{-x dt})/(x dt),  sigma_res = √((1-e^{-2x dt})/(2x) - cov_c² dt).
  - Antithetic: same variance paths (shared dW_j, eps_j), negated perpendicular
    stock BMs (dW_j_perp).

The simulator can also return a positivity probe: the running minimum of the
*raw* (pre-truncation) V^(1), V^(2) over all (path, step) pairs, the fraction of
pairs where each raw block variance is negative, and the Monte Carlo estimate of
E[V^(j)_T] (from the raw final factor sums, the unbiased estimator of g0^(j)(T)).
"""

from __future__ import annotations

import numpy as np

from .params import SymmetricTwoFactorParams
from ..common.forward_variance import ForwardVariance


def _ou_constants(x: np.ndarray, dt: float):
    """Exact-OU step constants for OU speeds x over a step dt."""
    alpha = np.exp(-x * dt)
    beta_sq = (1.0 - np.exp(-2.0 * x * dt)) / (2.0 * x)
    cov_c = (1.0 - alpha) / (x * dt)
    sigma_res = np.sqrt(np.maximum(beta_sq - cov_c ** 2 * dt, 0.0))
    return alpha, cov_c, sigma_res


def simulate_symmetric(
    params: SymmetricTwoFactorParams,
    market_xi0: ForwardVariance,
    S0: float,
    T: float,
    M_paths: int = 100_000,
    n_steps_per_year: int = 500,
    antithetic: bool = True,
    seed: int = 42,
    return_diag: bool = False,
):
    """Simulate log(S_T/S_0). If return_diag, also return a positivity-probe dict.

    Returns
    -------
    log_return : (M_paths,) float64
    diag (only if return_diag) : dict with keys
        min_V1_raw, min_V2_raw  — running min of raw block variance over (path,step)
        frac_neg1, frac_neg2    — fraction of (path,step) pairs with raw V^(j) < 0
        EV1_T, EV2_T            — MC mean of raw V^(j)_T (≈ g0^(j)(T))
        n_pairs                 — number of (path, step) pairs probed
    """
    rng = np.random.default_rng(seed)
    n_steps = max(1, int(round(T * n_steps_per_year)))
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    scale1, scale2 = params.forcing_weights()
    c1, x1 = params.c1, params.x1
    c2, x2 = params.c2, params.x2
    n1, n2 = len(c1), len(c2)

    M_base = (M_paths + 1) // 2 if antithetic else M_paths

    rho1_perp = np.sqrt(max(0.0, 1.0 - params.rho1 ** 2))
    rho2_perp = np.sqrt(max(0.0, 1.0 - params.rho2 ** 2))

    a1, cov1, sres1 = _ou_constants(x1, dt)
    a2, cov2, sres2 = _ou_constants(x2, dt)

    U1 = np.zeros((M_base, n1))
    U2 = np.zeros((M_base, n2))
    log_S_fwd = np.zeros(M_base)
    log_S_anti = np.zeros(M_base) if antithetic else None

    min_V1 = np.inf
    min_V2 = np.inf
    neg1 = 0
    neg2 = 0
    n_pairs = 0

    t = 0.0
    for _ in range(n_steps):
        dW1 = rng.standard_normal((M_base,)) * sqdt
        dW1_perp = rng.standard_normal((M_base,)) * sqdt
        dW2 = rng.standard_normal((M_base,)) * sqdt
        dW2_perp = rng.standard_normal((M_base,)) * sqdt
        eps_U1 = rng.standard_normal((M_base, n1))
        eps_U2 = rng.standard_normal((M_base, n2))

        xi_t = float(market_xi0(t))
        g1_t = scale1 * xi_t
        g2_t = scale2 * xi_t
        V1_raw = g1_t + U1 @ c1
        V2_raw = g2_t + U2 @ c2

        if return_diag:
            min_V1 = min(min_V1, float(V1_raw.min()))
            min_V2 = min(min_V2, float(V2_raw.min()))
            neg1 += int(np.count_nonzero(V1_raw < 0.0))
            neg2 += int(np.count_nonzero(V2_raw < 0.0))
            n_pairs += M_base

        V1 = np.maximum(V1_raw, 0.0)
        V2 = np.maximum(V2_raw, 0.0)
        sqrt_V1 = np.sqrt(V1)
        sqrt_V2 = np.sqrt(V2)

        sv1 = params.nu1 * sqrt_V1
        sv2 = params.nu2 * sqrt_V2
        U1 = (U1 * a1[None, :]
              + sv1[:, None] * (cov1[None, :] * dW1[:, None] + sres1[None, :] * eps_U1))
        U2 = (U2 * a2[None, :]
              + sv2[:, None] * (cov2[None, :] * dW2[:, None] + sres2[None, :] * eps_U2))

        drift = -0.5 * (V1 + V2) * dt
        dB1_fwd = params.rho1 * dW1 + rho1_perp * dW1_perp
        dB2_fwd = params.rho2 * dW2 + rho2_perp * dW2_perp
        log_S_fwd += drift + sqrt_V1 * dB1_fwd + sqrt_V2 * dB2_fwd

        if antithetic:
            dB1_anti = params.rho1 * dW1 - rho1_perp * dW1_perp
            dB2_anti = params.rho2 * dW2 - rho2_perp * dW2_perp
            log_S_anti += drift + sqrt_V1 * dB1_anti + sqrt_V2 * dB2_anti

        t += dt

    if antithetic:
        log_S = np.concatenate([log_S_fwd, log_S_anti])[:M_paths]
    else:
        log_S = log_S_fwd

    if not return_diag:
        return log_S

    xi_T = float(market_xi0(T))
    V1_T_raw = scale1 * xi_T + U1 @ c1
    V2_T_raw = scale2 * xi_T + U2 @ c2
    diag = dict(
        min_V1_raw=float(min_V1),
        min_V2_raw=float(min_V2),
        frac_neg1=float(neg1 / n_pairs) if n_pairs else 0.0,
        frac_neg2=float(neg2 / n_pairs) if n_pairs else 0.0,
        EV1_T=float(V1_T_raw.mean()),
        EV2_T=float(V2_T_raw.mean()),
        n_pairs=int(n_pairs),
    )
    return log_S, diag


def mc_call_prices(
    params: SymmetricTwoFactorParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    M_paths: int = 100_000,
    n_steps_per_year: int = 500,
    antithetic: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo call prices and standard errors at given strikes."""
    log_returns = simulate_symmetric(
        params, market_xi0, S0, T, M_paths, n_steps_per_year, antithetic, seed
    )
    S_T = S0 * np.exp(log_returns)
    strikes = np.asarray(strikes, dtype=float)

    prices = np.empty(len(strikes))
    stderr = np.empty(len(strikes))
    for j, K in enumerate(strikes):
        payoff = np.maximum(S_T - K, 0.0)
        prices[j] = payoff.mean()
        stderr[j] = payoff.std() / np.sqrt(len(payoff))
    return prices, stderr
