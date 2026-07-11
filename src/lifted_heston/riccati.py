"""Riccati ODE system for the lifted Heston characteristic function.

Implements the explicit-implicit scheme (Abi Jaber 2019 Appendix C, eq. C.2):

    ψ^{n,i}_0 = 0
                       1
    ψ^{n,i}_{k+1} = ───────── [ ψ^{n,i}_k + Δt F(u, Σ_j c_j ψ^{n,j}_k) ]
                   1 + x_i Δt

where (with λ = 0 in the forward-variance form)

    F(u, v) = (u² - u) / 2 + ρ ν u v + ν² v² / 2.

The system is stiff (x_n grows with n), and the explicit-implicit scheme is
unconditionally stable. We do *not* fall back to scipy.integrate.solve_ivp
because the stiff handling there has poor performance for this problem.

The function returns ψ^{n,i}(t_k) for all k on a uniform time grid, vectorised
over a batch of complex frequencies u.
"""

from __future__ import annotations

import numpy as np


def riccati_F(u: np.ndarray, v: np.ndarray, nu: float, rho: float) -> np.ndarray:
    """F(u, v) = 0.5 (u² - u) + ρ ν u v + 0.5 ν² v². Vectorised over u and v.

    With λ = 0, the canonical Riccati function in the forward-variance form.
    """
    return 0.5 * (u ** 2 - u) + rho * nu * u * v + 0.5 * nu ** 2 * v ** 2


def solve_riccati(
    u_grid: np.ndarray,           # shape (N_u,), complex
    c: np.ndarray,                # shape (n,), positive
    x: np.ndarray,                # shape (n,), positive ascending
    nu: float,
    rho: float,
    T: float,
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Riccati system on [0, T] for each u in u_grid.

    Returns
    -------
    t_grid : shape (n_steps + 1,), real, t_0 = 0, t_{n_steps} = T.
    psi    : shape (N_u, n, n_steps + 1), complex.

    Notes
    -----
    Memory: psi takes 16 * N_u * n * n_steps bytes (complex128).
    For N_u = 256, n = 20, n_steps = 200: ~16 MB. Fine.

    For the φ integral downstream, callers usually need the contracted state
    Ψ_k := Σ_j c_j ψ^{n,j}_k. Returning the full ψ tensor lets callers post-process.
    """
    N_u = len(u_grid)
    n = len(c)
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    psi = np.zeros((N_u, n, n_steps + 1), dtype=complex)
    # Pre-compute denominator (time-independent for uniform step)
    denom = 1.0 + x[None, :] * dt   # shape (1, n) — broadcasts over N_u

    for k in range(n_steps):
        v_k = (c[None, :] * psi[:, :, k]).sum(axis=1)   # (N_u,)
        F_k = riccati_F(u_grid, v_k, nu, rho)            # (N_u,)
        psi[:, :, k + 1] = (psi[:, :, k] + dt * F_k[:, None]) / denom

    return t_grid, psi
