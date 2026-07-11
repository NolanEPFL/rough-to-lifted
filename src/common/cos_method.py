"""Fang-Oosterlee COS method for European call pricing from a characteristic function.

Reference
---------
Fang, F., & Oosterlee, C. W. (2009). A novel pricing method for European options
based on Fourier-cosine series expansions. SIAM J. Sci. Comp. 31(2), 826–848.

We use the parametrisation in CONTEXT.md Section 2.6, with the truncation interval
[a, b] = [-L0 √(T ξ_0(T)), +L0 √(T ξ_0(T))], L0 = 12, N_cos = 256 by default.

The characteristic function passed in is for log(S_T / S_0), i.e. the "centered"
characteristic function. The user's lifted_heston.characteristic_function module
must return exp(φ^n(0, T)) WITHOUT the u log(S_0) prefactor.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def _chi_coeffs(c: float, d: float, a: float, omega: np.ndarray) -> np.ndarray:
    """Fourier-cosine coefficients of e^x on [c, d] w.r.t. basis cos(ω_j(x-a)).

    χ_j(c, d) = ∫_c^d e^x cos(ω_j(x-a)) dx  (closed form from CONTEXT.md §2.6).
    """
    result = np.empty(len(omega))
    result[0] = np.exp(d) - np.exp(c)          # limit as ω→0
    if len(omega) > 1:
        om = omega[1:]
        denom = 1.0 + om ** 2
        ed, ec = np.exp(d), np.exp(c)
        result[1:] = (
            np.cos(om * (d - a)) * ed - np.cos(om * (c - a)) * ec
            + om * (np.sin(om * (d - a)) * ed - np.sin(om * (c - a)) * ec)
        ) / denom
    return result


def _psi_coeffs(c: float, d: float, a: float, omega: np.ndarray) -> np.ndarray:
    """Fourier-cosine coefficients of 1 on [c, d] w.r.t. basis cos(ω_j(x-a)).

    ψ_j(c, d) = ∫_c^d cos(ω_j(x-a)) dx  (closed form from CONTEXT.md §2.6).
    """
    result = np.empty(len(omega))
    result[0] = d - c
    if len(omega) > 1:
        om = omega[1:]
        result[1:] = (np.sin(om * (d - a)) - np.sin(om * (c - a))) / om
    return result


def cos_call_prices(
    cf_centered: Callable[[np.ndarray], np.ndarray],
    S0: float,
    strikes: np.ndarray,
    T: float,
    a: float,
    b: float,
    N_cos: int = 256,
) -> np.ndarray:
    """Price a vector of European calls via the COS method.

    Parameters
    ----------
    cf_centered : characteristic function of log(S_T / S_0) at maturity T,
                  callable on a 1d numpy array of complex frequencies.
    S0 : spot price (= forward, since rates are zero).
    strikes : 1d array of strikes.
    T : maturity in years.
    a, b : truncation interval [a, b] for log(S_T / S_0).
    N_cos : number of cosine terms.

    Returns
    -------
    1d array of call prices, same length as strikes.

    Notes
    -----
    Let y = log(S_T/K).  The call payoff is K(e^y - 1)^+ for y > 0.
    The COS formula (Fang & Oosterlee 2009) in our notation:

        C(K) = (2K/(b-a)) Σ' Re[cf_y(ω_j) e^{-iω_j a}] (χ_j(0,b) - ψ_j(0,b))

    where cf_y(ω) = cf_centered(iω) · (S_0/K)^{iω}  is the CF of y = log(S_T/K).
    Substituting:

        C(K) = (2K/(b-a)) Σ' Re[cf_centered(iω_j) e^{-iω_j(a+k)}] (χ_j-ψ_j)

    with k = log(K/S_0).  The Σ' gives weight 1/2 to the j=0 term.
    Vectorised over strikes by broadcasting.
    """
    strikes = np.asarray(strikes, dtype=float)
    omega = np.arange(N_cos, dtype=float) * np.pi / (b - a)   # (N_cos,)

    # Evaluate CF at imaginary frequencies 1j*ω_j
    cf_vals = cf_centered(1j * omega)    # (N_cos,), complex

    # Payoff coefficients: independent of strike (integrate y from 0 to b)
    V_j = _chi_coeffs(0.0, b, a, omega) - _psi_coeffs(0.0, b, a, omega)  # (N_cos,)

    # Log-moneyness for each strike: k = log(K/S_0)
    k = np.log(strikes / S0)                                   # (M,)

    # Phase factor: accounts for shift from x=log(S_T/S_0) to y=log(S_T/K)
    # phase[m, j] = exp(-i ω_j (a + k_m))
    phase = np.exp(-1j * omega[None, :] * (a + k[:, None]))    # (M, N_cos)

    # COS sum: Re[ cf_vals[j] * phase[m,j] ] with half-weight on j=0
    cos_mat = np.real(cf_vals[None, :] * phase)                # (M, N_cos)
    cos_mat[:, 0] *= 0.5

    prices = (2.0 / (b - a)) * strikes * (cos_mat @ V_j)      # (M,)
    return prices


def cos_truncation_interval(T: float, total_var: float, L0: float = 12.0) -> tuple[float, float]:
    """Default truncation interval based on integrated forward variance."""
    sigma_total = np.sqrt(total_var)
    return (-L0 * sigma_total, L0 * sigma_total)
