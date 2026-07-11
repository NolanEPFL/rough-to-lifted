"""Symmetric-split two-factor lifted Heston: params → call prices → IV surface.

Numerics identical to the thesis: COS with N_cos=256, L0=12, truncation
[a,b]=[-L0 σ_tot, +L0 σ_tot] with σ_tot² = ∫_0^T ξ0(s) ds = market_xi0.integrated(T)
(the additive split sums back to the full ξ0, so the integrated variance — hence
the truncation — is the same as for the single-factor models). Riccati n_steps=200
for calibration, 1600 for the final reprice used in plots.
"""

from __future__ import annotations

import numpy as np

from .params import SymmetricTwoFactorParams
from .characteristic_function import SymmetricTwoFactorCF
from ..common.forward_variance import ForwardVariance
from ..common.cos_method import cos_call_prices, cos_truncation_interval
from ..common.black_scholes import bs_implied_vol


def symmetric_call_prices(
    params: SymmetricTwoFactorParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> np.ndarray:
    """European call prices via the symmetric two-factor CF + COS method."""
    cf = SymmetricTwoFactorCF(params, market_xi0, T, n_steps)
    total_var = market_xi0.integrated(T)
    a, b = cos_truncation_interval(T, total_var, L0)
    prices = cos_call_prices(
        cf.cf_centered, S0, np.asarray(strikes, dtype=float), T, a, b, N_cos
    )
    return np.maximum(prices, 0.0)


def symmetric_iv_surface(
    params: SymmetricTwoFactorParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes_per_T: dict[float, np.ndarray],
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> dict[float, np.ndarray]:
    """Full implied-volatility surface. Returns {T -> array of implied vols}."""
    iv: dict[float, np.ndarray] = {}
    for T_m, K_arr in strikes_per_T.items():
        K_arr = np.asarray(K_arr, dtype=float)
        prices = symmetric_call_prices(
            params, market_xi0, S0, K_arr, T_m, N_cos, L0, n_steps
        )
        ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), T_m, option_type="C")
            for p, K in zip(prices, K_arr)
        ])
        iv[float(T_m)] = ivs
    return iv
