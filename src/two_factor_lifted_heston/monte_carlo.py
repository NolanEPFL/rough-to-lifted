"""Euler-Maruyama simulator for the two-factor lifted Heston model.

Used only for the Monte Carlo sanity check (Section 8.7, check 3).
Not intended for production calibration.

SDE system (Definition 8.1, Case I):

    dS_t / S_t = sqrt(V^(1)_t) dB^(1)_t + sqrt(V^(2)_t) dB^(2)_t

    V^(1)_t = xi0^(1)(t) + sum_i c_i U^(e1,i)_t

    dU^(e1,i)_t = -x_i U^(e1,i)_t dt + nu1 * sqrt(V^(1)_t) dW^1_t   (lam1=0)

    dV^(2)_t = lam2 * (theta2 - V^(2)_t) dt + nu2 * sqrt(V^(2)_t) dW^2_t

with B^(j) = rho_j W^j + sqrt(1 - rho_j^2) W^{j,perp}.
(W_1, W_{1,perp}, W_2, W_{2,perp}) mutually independent.

Numerical choices:
  - Full truncation (replace negative V^(j) with 0 before sqrt).
  - Exact conditional OU simulation for U^(e1,i) factors: given V1_k constant over
    the step, the OU noise integral ∫ exp(-x*(T-s)) dW_s is decomposed into a part
    correlated with dW1_k and an independent residual (see technical notes below).
    This removes the large discretization bias of the Euler/implicit-Euler schemes
    for fast OU factors (x_i * dt >> 1 when H=0.1, n=20).
  - Antithetic variates: same variance paths (U1 driven by same dW1 and residual
    noise eps), negated perpendicular stock BMs (dW1_perp, dW2_perp). This is the
    conditional antithetic: given the variance path, the stock is Gaussian, and
    negating the independent BM component is a valid antithetic.
  - Pre-step V2 for the stock drift/diffusion to preserve the martingale property.

Exact OU step derivation:
  For dU = -x*U*dt + sigma*dW the exact solution over [t_k, t_{k+1}] is:
    U_{k+1} = U_k * exp(-x*dt) + sigma * ∫ exp(-x*(T-s)) dW_s
  The stochastic integral I_k = ∫ exp(-x*(T-s)) dW_s satisfies:
    E[I_k] = 0,  Var[I_k] = beta^2 = (1-exp(-2x*dt))/(2x)
    Cov(dW1_k, I_k) = (1-exp(-x*dt))/x
  Decomposing I_k = cov_coeff * dW1_k + sigma_res * eps, eps ~ N(0,1) independent:
    cov_coeff = (1-exp(-x*dt)) / (x*dt)
    sigma_res = sqrt(beta^2 - cov_coeff^2 * dt)
"""

from __future__ import annotations

import numpy as np

from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.common.forward_variance import ForwardVariance


def simulate_two_factor(
    params: TwoFactorLiftedHestonParams,
    market_xi0: ForwardVariance,
    S0: float,
    T: float,
    M_paths: int = 100_000,
    n_steps_per_year: int = 252,
    antithetic: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Simulate log(S_T / S_0) for M_paths paths.

    Returns log_return : (M_paths,) float64.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(1, int(round(T * n_steps_per_year)))
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    n1 = params.n1
    c, x = params.c, params.x

    M_base = (M_paths + 1) // 2 if antithetic else M_paths

    rho1_perp = np.sqrt(max(0.0, 1.0 - params.rho1 ** 2))
    rho2_perp = np.sqrt(max(0.0, 1.0 - params.rho2 ** 2))

    # Precompute per-factor exact-OU constants (once before the loop)
    alpha    = np.exp(-x * dt)                                    # (n1,) decay
    beta_sq  = (1.0 - np.exp(-2.0 * x * dt)) / (2.0 * x)        # (n1,) OU noise variance
    cov_c    = (1.0 - alpha) / (x * dt)                          # (n1,) cov(dW1,noise)/dt
    sigma_res = np.sqrt(np.maximum(beta_sq - cov_c ** 2 * dt, 0.0))  # (n1,) residual std

    U1        = np.zeros((M_base, n1))
    log_S_fwd  = np.zeros(M_base)
    log_S_anti = np.zeros(M_base) if antithetic else None
    V2 = np.full(M_base, params.V2_0)

    t = 0.0
    for _ in range(n_steps):
        dW1      = rng.standard_normal((M_base,)) * sqdt
        dW1_perp = rng.standard_normal((M_base,)) * sqdt
        dW2      = rng.standard_normal((M_base,)) * sqdt
        dW2_perp = rng.standard_normal((M_base,)) * sqdt
        eps_U1   = rng.standard_normal((M_base, n1))              # residual OU noise

        # Block-1 forward variance: xi0^(1)(t) = xi_market(t) - E[V^(2)_t]
        xi1_t = float(market_xi0(t)) - float(params.block2_mean(t))
        xi1_t = max(xi1_t, 0.0)
        V1 = np.maximum(xi1_t + U1 @ c, 0.0)                     # (M_base,)

        sqrt_V1 = np.sqrt(V1)
        sv1 = params.nu1 * sqrt_V1                                # (M_base,)

        # Exact conditional OU update (handles fast factors without bias)
        # U_{k+1}^i = U_k^i * alpha_i + sv1 * (cov_c_i * dW1 + sigma_res_i * eps_i)
        U1 = (U1 * alpha[None, :]
              + sv1[:, None] * (cov_c[None, :] * dW1[:, None]
                                + sigma_res[None, :] * eps_U1))

        # Block-2 CIR: pre-step V2_trunc for stock to preserve the martingale property
        V2_trunc = np.maximum(V2, 0.0)
        V2 = (V2 + params.lam2 * (params.theta2 - V2_trunc) * dt
              + params.nu2 * np.sqrt(V2_trunc) * dW2)
        V2 = np.maximum(V2, 0.0)

        sqrt_V2 = np.sqrt(V2_trunc)           # pre-step V2
        drift = -0.5 * (V1 + V2_trunc) * dt  # pre-step V2

        # Forward path
        dB1_fwd = params.rho1 * dW1 + rho1_perp * dW1_perp
        dB2_fwd = params.rho2 * dW2 + rho2_perp * dW2_perp
        log_S_fwd += drift + sqrt_V1 * dB1_fwd + sqrt_V2 * dB2_fwd

        if antithetic:
            # Antithetic: same (U1, V2) paths, negate perp BMs only
            dB1_anti = params.rho1 * dW1 - rho1_perp * dW1_perp
            dB2_anti = params.rho2 * dW2 - rho2_perp * dW2_perp
            log_S_anti += drift + sqrt_V1 * dB1_anti + sqrt_V2 * dB2_anti

        t += dt

    if antithetic:
        log_S = np.concatenate([log_S_fwd, log_S_anti])[:M_paths]
    else:
        log_S = log_S_fwd

    return log_S


def mc_call_prices(
    params: TwoFactorLiftedHestonParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    M_paths: int = 100_000,
    n_steps_per_year: int = 252,
    antithetic: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo call prices and standard errors at given strikes.

    Returns
    -------
    prices : (K,) float64
    stderr : (K,) float64  — Monte Carlo standard error per price.
    """
    log_returns = simulate_two_factor(
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
