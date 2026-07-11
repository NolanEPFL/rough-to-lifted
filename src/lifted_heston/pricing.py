"""High-level lifted Heston pricing: from parameters to implied volatility surface."""

from __future__ import annotations

import numpy as np

from .params import LiftedHestonParams
from .characteristic_function import LiftedHestonCharacteristicFunction
from ..common.forward_variance import ForwardVariance
from ..common.cos_method import cos_call_prices, cos_truncation_interval
from ..common.black_scholes import bs_implied_vol


def lifted_heston_call_prices(
    params: LiftedHestonParams,
    forward_variance: ForwardVariance,
    S0: float,
    strikes: np.ndarray,
    T: float,
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> np.ndarray:
    """Price European calls at maturity T on a strike grid via Riccati + COS.

    Returns
    -------
    call_prices : 1d array, same length as strikes.
    """
    cf = LiftedHestonCharacteristicFunction(params, forward_variance, T, n_steps)
    total_var = forward_variance.integrated(T)
    a, b = cos_truncation_interval(T, total_var, L0)
    prices = cos_call_prices(cf.cf_centered, S0, np.asarray(strikes, dtype=float),
                             T, a, b, N_cos)
    return np.maximum(prices, 0.0)   # clip rounding artefacts


def lifted_heston_iv_surface(
    params: LiftedHestonParams,
    forward_variance: ForwardVariance,
    S0: float,
    strikes_per_T: dict[float, np.ndarray] | list[tuple[float, np.ndarray]],
    N_cos: int = 256,
    L0: float = 12.0,
    n_steps: int = 200,
) -> dict[float, np.ndarray]:
    """Compute the full implied volatility surface.

    Parameters
    ----------
    strikes_per_T : maps each maturity T to its strike grid.

    Returns
    -------
    iv : dict {T -> array of implied volatilities}.

    Notes
    -----
    For each T, this builds one characteristic function (the dominant cost) and
    uses it for all strikes at that T. The Riccati ODE for different T values
    cannot be reused (since terminal time matters for the integration), so each T
    is independent. Trivially parallelisable across T if needed.
    """
    items = strikes_per_T.items() if isinstance(strikes_per_T, dict) else strikes_per_T
    iv: dict[float, np.ndarray] = {}
    for T_m, K_arr in items:
        K_arr = np.asarray(K_arr, dtype=float)
        prices = lifted_heston_call_prices(
            params, forward_variance, S0, K_arr, T_m, N_cos, L0, n_steps
        )
        ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), T_m, option_type="C")
            for p, K in zip(prices, K_arr)
        ])
        iv[float(T_m)] = ivs
    return iv
