"""Fractional Riccati equation for rough Heston (El Euch & Rosenbaum 2018).

The equation is

    g(u, t) = (1/Γ(α)) ∫_0^t (t - s)^{α-1} F(u, g(u, s)) ds,   α = H + 1/2,

where F is the same Riccati function as in lifted Heston with λ = 0:

    F(u, v) = (u² - u)/2 + ρ ν u v + ν² v² / 2.

We solve by the Adams predictor–corrector (P-C) scheme derived for Volterra
integral equations with weakly singular kernels (Diethelm-Ford type).

Predictor (Adams-Bashforth, exact product-integration with piecewise-constant f):
    g^P_{n+1} = (h^α/Γ(α+1)) * Σ_{j=0}^n [(n+1-j)^α - (n-j)^α] * F_j

Corrector (replace last sub-interval [t_n, t_{n+1}] with linear interpolation):
    g_{n+1} = g^P_{n+1} + (h^α/Γ(α+2)) * (F^P_{n+1} − F_n)

The corrector weight follows from exact integration of the kernel with a
piecewise-linear approximation on the last sub-interval.

Complexity: O(N_u × N²) per surface (N = n_steps). With N=1000 and N_u=256
this is ~2.5 × 10^8 fused multiply-adds — runs in seconds with numpy.

This is ONLY used in scripts/01_lh_convergence.py as a ground-truth reference.
If it is too slow or numerically unstable, replace with n=500 lifted Heston.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as Gamma


def solve_fractional_riccati(
    u_grid: np.ndarray,    # complex, shape (N_u,)
    H: float,
    nu: float,
    rho: float,
    T: float,
    n_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the fractional Riccati for rough Heston using Adams P-C.

    Parameters
    ----------
    u_grid  : complex frequencies, shape (N_u,).
    H       : Hurst index ∈ (0, 0.5).
    nu, rho : rough Heston parameters.
    T       : terminal time.
    n_steps : number of uniform time steps.

    Returns
    -------
    t_grid : real, shape (n_steps + 1,), t_0=0, t_{n_steps}=T.
    g      : complex, shape (N_u, n_steps + 1).
              g[:, k] = g(u_grid, t_k).
    """
    from ..lifted_heston.riccati import riccati_F

    alpha  = H + 0.5
    h      = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)
    N_u    = len(u_grid)

    # Pre-computed power table: p[k] = k^alpha, k = 0, ..., n_steps+1
    p = np.arange(n_steps + 2, dtype=float) ** alpha   # p[0] = 0 for α ∈ (0,1)

    coef1 = h ** alpha / Gamma(alpha + 1)   # predictor coefficient
    coef2 = h ** alpha / Gamma(alpha + 2)   # corrector correction coefficient

    g        = np.zeros((N_u, n_steps + 1), dtype=complex)
    F_stored = np.zeros((N_u, n_steps + 1), dtype=complex)

    # g[:, 0] = 0  (initial condition)
    F_stored[:, 0] = riccati_F(u_grid, g[:, 0], nu, rho)  # F(u, 0) = 0.5*(u^2 - u)

    for n in range(n_steps):
        j = np.arange(n + 1)                              # j = 0, ..., n
        w = p[n + 1 - j] - p[n - j]                       # (n+1,) predictor weights

        # Predictor: Adams-Bashforth (product integration, piecewise-constant f)
        g_pred = coef1 * (F_stored[:, :n + 1] * w[None, :]).sum(axis=1)
        F_pred = riccati_F(u_grid, g_pred, nu, rho)

        # Corrector: improve the last sub-interval with linear interpolation
        # Exact kernel integral on [t_n, t_{n+1}] with f linear → f^P at right end:
        #   h^α/Γ(α+2) * (F^P − F_n)  replaces the rectangle-rule error.
        g[:, n + 1] = g_pred + coef2 * (F_pred - F_stored[:, n])
        F_stored[:, n + 1] = riccati_F(u_grid, g[:, n + 1], nu, rho)

    return t_grid, g
