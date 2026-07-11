"""Rough kernel K_H and finite-dimensional approximation K_n.

K_H(t)  = t^{H - 1/2} / Γ(H + 1/2),       t > 0
K_n(t)  = Σ_{i=1}^n c_i e^{-x_i t}

Used by both lifted_heston (with geometric grid) and abergomi (with L²-fit grid).

Note that K_H is singular at t = 0 for H < 1/2; never evaluate K_H at t = 0
without an explicit ε.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as Gamma_fn


def K_H(t: float | np.ndarray, H: float) -> float | np.ndarray:
    """Rough Riemann-Liouville kernel K_H(t) = t^{H-1/2} / Γ(H + 1/2)."""
    return np.power(np.asarray(t, dtype=float), H - 0.5) / Gamma_fn(H + 0.5)


def K_n(t: float | np.ndarray, c: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """K_n(t) = Σ c_i exp(-x_i t). Vectorised in t."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    result = np.exp(-x[None, :] * t[:, None]) @ c   # (T,)
    return float(result[0]) if scalar else result


def kernel_l2_error(
    H: float, c: np.ndarray, x: np.ndarray,
    eps: float = 1.0 / 365.0, T_max: float = 2.0,
    n_quad: int = 200, log_grid: bool = True,
) -> float:
    """Approximate ‖K_H - K_n‖_{L²([eps, T_max])} via quadrature.

    Use a log-spaced grid by default since K_H is concentrated near t = eps.
    """
    if log_grid:
        t = np.logspace(np.log10(eps), np.log10(T_max), n_quad)
    else:
        t = np.linspace(eps, T_max, n_quad)
    diff = K_H(t, H) - K_n(t, c, x)
    return float(np.sqrt(np.trapezoid(diff ** 2, t)))
