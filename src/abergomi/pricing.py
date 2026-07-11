"""aBergomi MC pricing: from simulated paths to call prices and IV surface.

For multi-maturity pricing a SINGLE simulation up to T_max is used, recording
log_S at each requested maturity.  This shares random numbers across maturities
(Common Random Numbers for calibration stability).
"""

from __future__ import annotations

import numpy as np

from .kernel_fit import ABergomiParams
from .simulation import simulate_abergomi
from ..common.forward_variance import ForwardVariance
from ..common.black_scholes import bs_implied_vol, bs_vega


def abergomi_call_prices(
    params: ABergomiParams,
    forward_variance: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    M_paths: int = 50_000,
    n_steps: int | None = None,
    antithetic: bool = True,
    qmc: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-maturity European call prices and MC standard errors.

    Returns
    -------
    prices  : mean call prices, shape (len(strikes),).
    std_err : MC standard errors (raw price), shape (len(strikes),).
    """
    if n_steps is None:
        n_steps = max(100, int(100 * T))

    strikes = np.asarray(strikes, dtype=float)
    _, log_S = simulate_abergomi(
        params, forward_variance, S0, T,
        M_paths=M_paths, n_steps=n_steps,
        antithetic=antithetic, qmc=qmc, seed=seed,
    )
    S_T   = np.exp(log_S[:, 0])   # (M_eff,)
    M_eff = len(S_T)

    payoffs  = np.maximum(S_T[:, None] - strikes[None, :], 0.0)  # (M_eff, K)
    prices   = payoffs.mean(axis=0)
    std_errs = payoffs.std(axis=0, ddof=1) / np.sqrt(M_eff)

    return prices, std_errs


def abergomi_iv_surface(
    params: ABergomiParams,
    forward_variance: ForwardVariance,
    S0: float,
    strikes_per_T: dict[float, np.ndarray] | list[tuple[float, np.ndarray]],
    M_paths: int = 50_000,
    n_steps_per_year: int = 100,
    antithetic: bool = True,
    qmc: bool = True,
    seed: int = 42,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Full implied-volatility surface via one MC simulation.

    A single call to simulate_abergomi reaches T_max and records log_S at
    every requested maturity.  This shares the same random numbers across
    maturities (CRN).

    Returns
    -------
    iv      : dict {T_actual → array of implied vols}.
    iv_std  : dict {T_actual → array of IV MC std-errors (via vega propagation)}.
    """
    items = sorted(
        strikes_per_T.items() if isinstance(strikes_per_T, dict) else strikes_per_T
    )
    maturities = np.array([T for T, _ in items])
    T_max      = float(maturities.max())
    n_steps    = max(100, int(n_steps_per_year * T_max))

    rec_times, log_S = simulate_abergomi(
        params, forward_variance, S0, T_max,
        M_paths=M_paths, n_steps=n_steps,
        antithetic=antithetic, qmc=qmc, seed=seed,
        record_at=maturities,
    )
    M_eff = log_S.shape[0]

    iv:     dict[float, np.ndarray] = {}
    iv_std: dict[float, np.ndarray] = {}

    for col, ((_, K_arr), T_act) in enumerate(zip(items, rec_times)):
        T_act  = float(T_act)
        K_arr  = np.asarray(K_arr, dtype=float)
        S_T    = np.exp(log_S[:, col])

        payoffs  = np.maximum(S_T[:, None] - K_arr[None, :], 0.0)
        prices   = payoffs.mean(axis=0)
        std_errs = payoffs.std(axis=0, ddof=1) / np.sqrt(M_eff)

        ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), T_act)
            for p, K in zip(prices, K_arr)
        ])

        vegas = np.array([
            bs_vega(S0, float(K), T_act, float(iv_v))
            if np.isfinite(iv_v) else np.nan
            for K, iv_v in zip(K_arr, ivs)
        ])
        iv_stds = np.where(
            np.isfinite(vegas) & (vegas > 0.0), std_errs / vegas, np.nan
        )

        iv[T_act]     = ivs
        iv_std[T_act] = iv_stds

    return iv, iv_std
