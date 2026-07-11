"""Black-Scholes pricing and implied volatility inversion.

We assume zero rates and zero dividends throughout (work in the forward measure).
All inputs are in absolute units; strikes K and forward F are positive reals,
maturities T in years, total variance sigma * sqrt(T).

References
----------
Standard textbook formulas; see e.g. Hull, *Options, Futures, and Other Derivatives*.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call_price(F: float | np.ndarray, K: float | np.ndarray,
                  T: float | np.ndarray, sigma: float | np.ndarray) -> float | np.ndarray:
    """Black-Scholes (Black-76) call price with zero rates/dividends.

    Parameters
    ----------
    F : forward price
    K : strike
    T : time to maturity in years
    sigma : Black-Scholes volatility (annualised)

    Returns
    -------
    Call price.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def bs_put_price(F: float | np.ndarray, K: float | np.ndarray,
                 T: float | np.ndarray, sigma: float | np.ndarray) -> float | np.ndarray:
    """Black-Scholes put price."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def bs_vega(F: float | np.ndarray, K: float | np.ndarray,
            T: float | np.ndarray, sigma: float | np.ndarray) -> float | np.ndarray:
    """Black-Scholes vega = ∂C/∂σ. Used for vega-weighted calibration."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return F * norm.pdf(d1) * sqrt_T


def bs_implied_vol(price: float, F: float, K: float, T: float,
                   option_type: str = "C",
                   sigma_min: float = 1e-6, sigma_max: float = 5.0) -> float:
    """Invert the Black-Scholes formula via Brent's method.

    Parameters
    ----------
    price : option mid price (must lie strictly between the no-arbitrage bounds).
    option_type : 'C' for call, 'P' for put.

    Returns
    -------
    Implied volatility (annualised). Returns np.nan if inversion fails (price out
    of bounds, bracket not bracketing a root, etc.). The caller is responsible for
    filtering out NaNs from the surface.
    """
    if option_type == "C":
        intrinsic = max(float(F) - float(K), 0.0)
        upper = float(F)
        def objective(sigma):
            return float(bs_call_price(F, K, T, sigma)) - price
    else:
        intrinsic = max(float(K) - float(F), 0.0)
        upper = float(K)
        def objective(sigma):
            return float(bs_put_price(F, K, T, sigma)) - price

    if price <= intrinsic or price >= upper:
        return np.nan
    try:
        return brentq(objective, sigma_min, sigma_max, xtol=1e-10, rtol=1e-10)
    except ValueError:
        return np.nan
