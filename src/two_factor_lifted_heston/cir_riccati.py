"""Scalar Riccati ODE for block 2 (classical CIR / Heston).

Theorem 8.7 of the thesis, eq. (8.13) and (8.16):

    dpsi2/ds = F_2(u, psi2(s)) - lam2 * psi2(s),    psi2(0) = 0

    F_2(u, w) = (u^2 - u)/2 + rho2 * nu2 * u * w + (nu2^2 / 2) * w^2

The -lam2*psi term is handled implicitly by the solver (denom = 1 + lam2*dt),
consistent with the lifted Heston Riccati in lifted_heston/riccati.py.
F_2 itself does NOT include the -lam2*w term to avoid double-counting.

The characteristic function contribution of block 2 is:

    Phi_T^(2)(u) = exp( V2_0 * psi2(T) + lam2 * theta2 * INT_0^T psi2(s) ds )

where the integral is approximated by the trapezoidal rule on the same uniform
time grid as the Riccati solver, mirroring the phi integral in block 1.

We use the same explicit-implicit scheme as src/lifted_heston/riccati.py
(eq. C.2 of Abi Jaber 2019). For the scalar case this reduces to:

    psi2_{k+1} = (psi2_k + dt * F_2(u, psi2_k)) / (1 + lam2 * dt)

The denominator 1 + lam2 * dt handles the stiff -lam2*psi part implicitly;
F_2 contributes only the nonlinear g(u, psi) part.
"""

from __future__ import annotations

import numpy as np


def cir_riccati_F(
    u: np.ndarray,   # (N_u,) complex
    w: np.ndarray,   # (N_u,) complex
    nu2: float,
    rho2: float,
) -> np.ndarray:
    """g(u, w) = (u^2-u)/2 + rho2*nu2*u*w + (nu2^2/2)*w^2.

    This is the nonlinear part of the CIR Riccati RHS.
    The -lam2*w linear term is handled implicitly by the solver.
    """
    return (
        0.5 * (u ** 2 - u)
        + rho2 * nu2 * u * w
        + 0.5 * nu2 ** 2 * w ** 2
    )


def solve_cir_riccati(
    u_grid: np.ndarray,   # (N_u,) complex
    nu2: float,
    rho2: float,
    lam2: float,
    T: float,
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the scalar CIR Riccati on [0, T] for each u in u_grid.

    Scheme: psi_{k+1} = (psi_k + dt * F_2(u, psi_k)) / (1 + lam2*dt)
    The -lam2*psi term is implicit (denominator), F_2 is explicit.

    Returns
    -------
    t_grid : (n_steps + 1,) real, t_0=0, t_{n_steps}=T.
    psi2   : (N_u, n_steps + 1) complex, psi2[:, 0] = 0.
    """
    N_u = len(u_grid)
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    psi2 = np.zeros((N_u, n_steps + 1), dtype=complex)
    denom = 1.0 + lam2 * dt   # implicit -lam2*psi treatment

    for k in range(n_steps):
        F_k = cir_riccati_F(u_grid, psi2[:, k], nu2, rho2)   # (N_u,)
        psi2[:, k + 1] = (psi2[:, k] + dt * F_k) / denom

    return t_grid, psi2


def cir_cf_centered(
    u_grid: np.ndarray,
    nu2: float,
    rho2: float,
    lam2: float,
    theta2: float,
    V2_0: float,
    T: float,
    n_steps: int = 200,
) -> np.ndarray:
    """Phi_T^(2)(u) = exp(V2_0 * psi2(T) + lam2*theta2 * INT psi2(s) ds).

    Returns (N_u,) complex array.
    """
    t_grid, psi2 = solve_cir_riccati(u_grid, nu2, rho2, lam2, T, n_steps)

    # Integral INT_0^T psi2(s) ds via trapezoidal rule
    integral = np.trapezoid(psi2, t_grid, axis=1)  # (N_u,)

    log_phi2 = V2_0 * psi2[:, -1] + lam2 * theta2 * integral
    result = np.exp(log_phi2)
    result = np.where(np.isfinite(result), result, 0.0 + 0.0j)
    return result
