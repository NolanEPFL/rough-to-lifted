"""Two-factor lifted Heston: parameter → call prices → IV surface."""

from __future__ import annotations

import numpy as np

from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.characteristic_function import TwoFactorLiftedHestonCF
from src.common.forward_variance import ForwardVariance
from src.common.cos_method import cos_call_prices, cos_truncation_interval
from src.common.black_scholes import bs_implied_vol


def two_factor_lh_call_prices(
    params: TwoFactorLiftedHestonParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> np.ndarray:
    """European call prices via the two-factor lifted Heston CF + COS method."""
    cf = TwoFactorLiftedHestonCF(params, market_xi0, T, n_steps)
    total_var = market_xi0.integrated(T)
    a, b = cos_truncation_interval(T, total_var, L0)
    prices = cos_call_prices(
        cf.cf_centered, S0, np.asarray(strikes, dtype=float), T, a, b, N_cos
    )
    return np.maximum(prices, 0.0)


def two_factor_lh_iv_surface(
    params: TwoFactorLiftedHestonParams,
    market_xi0: ForwardVariance,
    S0: float,
    strikes_per_T: dict[float, np.ndarray],
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> dict[float, np.ndarray]:
    """Compute the full implied volatility surface.

    Returns dict {T -> array of implied volatilities}.
    """
    iv: dict[float, np.ndarray] = {}
    for T_m, K_arr in strikes_per_T.items():
        K_arr = np.asarray(K_arr, dtype=float)
        prices = two_factor_lh_call_prices(
            params, market_xi0, S0, K_arr, T_m, N_cos, L0, n_steps
        )
        ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), T_m, option_type="C")
            for p, K in zip(prices, K_arr)
        ])
        iv[float(T_m)] = ivs
    return iv
