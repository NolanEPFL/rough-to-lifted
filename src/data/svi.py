"""Raw SVI smile fitting (Gatheral 2004) — per-slice calibration.

Parametrisation
---------------
    w(k; a, b, rho, m, s) = a + b * [rho*(k - m) + sqrt((k - m)^2 + s^2)]

where k = log(K/F) is log-moneyness and w(k) = sigma_impl(k)^2 * T is total
implied variance.  's' is the SVI smoothness parameter (not implied vol).

ATM quantities
--------------
    w(0)       = a + b * (-rho*m + sqrt(m^2 + s^2))
    sigma_ATM  = sqrt(w(0) / T)

    d sigma / dk |_{k=0} = w'(0) / (2 * sigma_ATM * T)
    w'(0)      = b * [rho - m / sqrt(m^2 + s^2)]

For SPX the ATM skew derivative is negative (downward skew).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_PENALTY_WEIGHT = 1e6   # quadratic penalty for a + b*s*sqrt(1-rho^2) < 0


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class SVISliceFit:
    """Raw SVI fit for one expiry slice.

    Fields
    ------
    a, b, rho, m, s : SVI parameters (see module docstring)
    T               : maturity in years
    n_points        : number of strikes used in the fit
    rmse_iv         : in-sample RMSE on sigma (vol points, not variance)
    """
    a: float
    b: float
    rho: float
    m: float
    s: float        # SVI smoothness (not implied vol)
    T: float
    n_points: int
    rmse_iv: float

    def total_var(self, k: np.ndarray) -> np.ndarray:
        """w(k) = a + b*[rho*(k-m) + sqrt((k-m)^2 + s^2)]."""
        k = np.asarray(k, dtype=float)
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d**2 + self.s**2))

    def iv(self, k: np.ndarray) -> np.ndarray:
        """Implied vol sigma(k) = sqrt(max(w(k), 0) / T)."""
        w = self.total_var(np.asarray(k, dtype=float))
        return np.sqrt(np.maximum(w, 0.0) / self.T)

    def atm_iv(self) -> float:
        """ATM implied vol: sigma_ATM = sqrt(w(0) / T).

        w(0) = a + b*(-rho*m + sqrt(m^2 + s^2))
        """
        w0 = self.a + self.b * (
            -self.rho * self.m + np.sqrt(self.m**2 + self.s**2)
        )
        return float(np.sqrt(max(w0, 0.0) / self.T))

    def atm_skew_derivative(self) -> float:
        """Signed d sigma_impl/dk evaluated at k = 0.

        Derived from sigma(k)^2 = w(k)/T by implicit differentiation:
            d sigma/dk|_{k=0} = w'(0) / (2 * sigma_ATM * T)
            w'(0) = b * [rho - m / sqrt(m^2 + s^2)]

        For SPX this is negative (downward skew).
        """
        w_prime_0 = self.b * (
            self.rho - self.m / np.sqrt(self.m**2 + self.s**2)
        )
        sigma_atm = self.atm_iv()
        if sigma_atm < 1e-10:
            return 0.0
        return float(w_prime_0 / (2.0 * sigma_atm * self.T))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _svi_w(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, s = params
    d = k - m
    return a + b * (rho * d + np.sqrt(d**2 + s**2))


def _objective(
    params: np.ndarray,
    k: np.ndarray,
    w_data: np.ndarray,
    fit_weights: np.ndarray,
) -> float:
    a, b, rho, m, s = params
    w_model = _svi_w(k, params)
    resid = w_model - w_data
    ssr = float(np.dot(fit_weights, resid**2))
    # Enforce a + b*s*sqrt(1 - rho^2) >= 0 via quadratic penalty
    min_var = a + b * s * np.sqrt(max(1.0 - rho**2, 0.0))
    penalty = _PENALTY_WEIGHT * max(0.0, -min_var) ** 2
    return ssr + penalty


# ── Public API ────────────────────────────────────────────────────────────────

def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: Optional[np.ndarray] = None,
) -> SVISliceFit:
    """Fit raw SVI to one smile slice in total-variance space.

    Fitting is done on w = iv^2 * T (total variance), not on IV directly,
    which gives better numerical conditioning (SVI literature convention).

    Default per-point weights: Gaussian centred at k=0 with bandwidth 0.10,
        w_i = exp(-(k_i / 0.10)^2)
    to emphasise the near-ATM region that controls the skew.

    Initialisation: 27-point multi-start grid
        m_0   in {-0.05, 0.0, +0.05}
        s_0   in {0.05, 0.10, 0.20}
        rho_0 in {-0.7, -0.3, 0.0}
        a_0   = min(w_data)
        b_0   = max(0.01, (max(w_data) - min(w_data)) / 0.2)
    Each start is optimised with L-BFGS-B; the best weighted SSR is kept.

    Parameter bounds:
        b  >= 0,  |rho| <= 1 - 1e-6,  s >= 1e-4,  m in [-1, 1]
    Positivity  a + b*s*sqrt(1-rho^2) >= 0  is enforced by a quadratic
    penalty (weight 1e6) added to the objective.

    Parameters
    ----------
    k       : log-moneyness k_i = log(K_i / F), shape (N,)
    iv      : implied vol, same shape
    T       : maturity in years
    weights : optional per-point weights (default: Gaussian as above)

    Returns
    -------
    SVISliceFit

    Raises
    ------
    ValueError if len(k) < 5 or all total-variance values are <= 0.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)

    if len(k) < 5:
        raise ValueError(f"Need at least 5 strikes for SVI fit, got {len(k)}.")

    w_data = iv**2 * T
    if np.all(w_data <= 0):
        raise ValueError("All total-variance values are non-positive.")

    # Normalised fit weights
    if weights is None:
        fit_weights = np.exp(-(k / 0.10) ** 2)
    else:
        fit_weights = np.asarray(weights, dtype=float).copy()
    fit_weights /= fit_weights.sum()

    w_min = float(np.min(w_data))
    w_max = float(np.max(w_data))
    a0 = w_min
    b0 = max(0.01, (w_max - w_min) / 0.2)

    bounds = [
        (None, None),                     # a
        (0.0, 10.0),                      # b >= 0
        (-1.0 + 1e-6, 1.0 - 1e-6),       # |rho| < 1
        (-1.0, 1.0),                      # m (practical)
        (1e-4, 1.0),                      # s >= 1e-4
    ]

    best_result = None
    best_val = np.inf

    for m0 in (-0.05, 0.0, 0.05):
        for s0 in (0.05, 0.10, 0.20):
            for rho0 in (-0.7, -0.3, 0.0):
                x0 = np.array([a0, b0, rho0, m0, s0])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = minimize(
                            _objective,
                            x0,
                            args=(k, w_data, fit_weights),
                            method="L-BFGS-B",
                            bounds=bounds,
                            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
                        )
                    if res.fun < best_val:
                        best_val = res.fun
                        best_result = res
                except Exception:
                    continue

    if best_result is None:
        raise ValueError("SVI optimisation failed for all starting points.")

    a, b, rho, m, s = best_result.x

    w_fit = _svi_w(k, best_result.x)
    iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
    rmse_iv = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))

    return SVISliceFit(
        a=float(a), b=float(b), rho=float(rho), m=float(m), s=float(s),
        T=float(T), n_points=int(len(k)), rmse_iv=rmse_iv,
    )


def fit_svi_surface(
    surface,
    min_points_per_slice: int = 5,
) -> dict[float, SVISliceFit]:
    """Fit raw SVI per maturity on a cleaned SPXSurface.

    Slices with fewer than min_points_per_slice valid strikes are skipped
    (a warning is logged, no exception is raised).

    Parameters
    ----------
    surface              : SPXSurface from src.data.spx_loader
    min_points_per_slice : minimum number of strikes required to fit

    Returns
    -------
    {T -> SVISliceFit} for all fitted maturities
    """
    mats = np.unique(surface.maturities)
    result: dict[float, SVISliceFit] = {}
    for T in mats:
        mask = surface.maturities == T
        Ks = surface.strikes[mask]
        Fs = surface.forwards[mask]
        iv = surface.iv_mkt[mask]
        k = np.log(Ks / Fs)
        fin = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
        k_clean, iv_clean = k[fin], iv[fin]
        if len(k_clean) < min_points_per_slice:
            logger.warning(
                "SVI: skipping T=%.4f — only %d valid strikes (need %d).",
                T, len(k_clean), min_points_per_slice,
            )
            continue
        try:
            result[float(T)] = fit_svi_slice(k_clean, iv_clean, float(T))
        except Exception as exc:
            logger.warning("SVI: fit failed for T=%.4f: %s", T, exc)
    return result


def fit_svi_from_iv_dict(
    iv_dict: dict[float, np.ndarray],
    K_dict: dict[float, np.ndarray],
    F_dict: dict[float, float],
    min_points_per_slice: int = 5,
) -> dict[float, SVISliceFit]:
    """Fit raw SVI from the dict format used by the plot scripts.

    Model IVs arrive as {T -> iv_array} with corresponding {T -> K_array}
    and {T -> forward}.

    Parameters
    ----------
    iv_dict : {T -> array of implied vols}
    K_dict  : {T -> array of absolute strikes K}
    F_dict  : {T -> forward price F_T}

    Returns
    -------
    {T -> SVISliceFit} for all successfully fitted maturities
    """
    result: dict[float, SVISliceFit] = {}
    for T in sorted(iv_dict.keys()):
        iv = np.asarray(iv_dict[T], dtype=float)
        Ks = K_dict.get(T)
        if Ks is None:
            continue
        Ks = np.asarray(Ks, dtype=float)
        F = float(F_dict.get(T, float(np.nanmedian(Ks))))
        k = np.log(Ks / F)
        fin = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
        k_clean, iv_clean = k[fin], iv[fin]
        if len(k_clean) < min_points_per_slice:
            logger.warning(
                "SVI: skipping T=%.4f — only %d valid strikes (need %d).",
                T, len(k_clean), min_points_per_slice,
            )
            continue
        try:
            result[float(T)] = fit_svi_slice(k_clean, iv_clean, float(T))
        except Exception as exc:
            logger.warning("SVI: fit failed for T=%.4f: %s", T, exc)
    return result
