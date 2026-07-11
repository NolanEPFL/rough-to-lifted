"""Rough Heston IV surface via fractional Riccati + COS pricing.

Pricing is structurally identical to lifted Heston (CONTEXT.md §2.4):

    Φ_T(u) = exp(φ(u, T)),  φ(u, T) = ∫_0^T F(u, g(u, s)) ξ_0(T-s) ds,

but g comes from the fractional Riccati (solve_fractional_riccati) rather than
the Markovian ODE system. Used ONLY as ground truth in Experiment 1.
"""

from __future__ import annotations

import numpy as np

from .fractional_riccati import solve_fractional_riccati
from ..lifted_heston.riccati import riccati_F
from ..common.forward_variance import ForwardVariance
from ..common.cos_method import cos_call_prices, cos_truncation_interval
from ..common.black_scholes import bs_implied_vol


def rough_heston_cf_centered(
    u_grid: np.ndarray,
    H: float,
    nu: float,
    rho: float,
    forward_variance: ForwardVariance,
    T: float,
    n_steps: int = 1000,
) -> np.ndarray:
    """Characteristic function of log(S_T/S_0) under rough Heston.

    Returns exp(φ(u, T)) where φ = ∫_0^T F(u, g(u,s)) ξ_0(T-s) ds.
    """
    t_grid, g = solve_fractional_riccati(u_grid, H, nu, rho, T, n_steps)

    # F_k = F(u_grid, g_k) for k = 0,...,n_steps  (same formula as lifted Heston)
    F_k = riccati_F(u_grid[:, None], g, nu, rho)           # (N_u, n_steps+1)

    xi_vals = np.asarray(forward_variance(T - t_grid), dtype=float)   # (n_steps+1,)
    phi = np.trapezoid(F_k * xi_vals[None, :], t_grid, axis=1)        # (N_u,)

    result = np.exp(phi)
    result = np.where(np.isfinite(result), result, 0.0 + 0.0j)
    return result


def rough_heston_iv_surface(
    H: float, nu: float, rho: float,
    forward_variance: ForwardVariance,
    S0: float,
    strikes_per_T: dict[float, np.ndarray],
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 1000,
) -> dict[float, np.ndarray]:
    """Compute rough Heston IV surface as ground-truth reference.

    For each maturity T, solves the fractional Riccati on [0, T] (O(N_u × N²)),
    then prices calls via COS and inverts to implied vols.
    """
    iv: dict[float, np.ndarray] = {}

    for T, K_arr in strikes_per_T.items():
        K_arr     = np.asarray(K_arr, dtype=float)
        total_var = forward_variance.integrated(float(T))
        a, b      = cos_truncation_interval(float(T), total_var, L0)

        def cf(u_grid):
            return rough_heston_cf_centered(u_grid, H, nu, rho, forward_variance,
                                            float(T), n_steps)

        prices = cos_call_prices(cf, S0, K_arr, float(T), a, b, N_cos)
        prices = np.maximum(prices, 0.0)

        ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), float(T))
            for p, K in zip(prices, K_arr)
        ])
        iv[float(T)] = ivs

    return iv
