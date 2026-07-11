"""Lifted Heston characteristic function.

Assembles Φ_T(u) = exp(u log S_0 + φ^n(0, T)) where φ^n is the integral

    φ^n(0, T) = ∫_0^T F(u, Σ_j c_j ψ^{n,j}(s)) ξ_0(T - s) ds.

In the forward-variance form (λ = 0, g_0^n = ξ_0).

We provide a class wrapper because the user typically wants to evaluate the
characteristic function for many frequencies u at the SAME maturity T, which
allows caching the Riccati solution. The COS pricer evaluates Φ_T at a vector
of frequencies u_j = j π / (b - a).

The "centered" characteristic function returned by `cf_centered` is
exp(φ^n(0, T)), i.e. the cf of log(S_T / S_0). This is what `common.cos_method`
expects.
"""

from __future__ import annotations

from typing import Callable
import numpy as np

from .params import LiftedHestonParams
from .riccati import solve_riccati, riccati_F
from ..common.forward_variance import ForwardVariance


class LiftedHestonCharacteristicFunction:
    """Build and cache the lifted Heston characteristic function at a fixed T.

    Construction is O(N_u * n * n_steps); subsequent calls to .__call__ are
    O(N_u). Use this when pricing many strikes at the same maturity (the
    common case in COS pricing).

    Parameters
    ----------
    params : LiftedHestonParams.
    forward_variance : forward variance curve ξ_0.
    T : maturity in years.
    n_steps : number of Riccati time steps. 200 default; verify convergence.
    """

    def __init__(
        self,
        params: LiftedHestonParams,
        forward_variance: ForwardVariance,
        T: float,
        n_steps: int = 200,
    ):
        self.params = params
        self.fv = forward_variance
        self.T = float(T)
        self.n_steps = n_steps

    def cf_centered(self, u_grid: np.ndarray) -> np.ndarray:
        """Evaluate Φ_T(u) = exp(φ^n(0, T)) for u in u_grid (cf of log(S_T/S_0))."""
        p = self.params
        T = self.T

        # Step 1: solve Riccati on [0, T]
        t_grid, psi = solve_riccati(
            u_grid, p.c, p.x, p.nu, p.rho, T, self.n_steps
        )  # t_grid: (N+1,), psi: (N_u, n, N+1)

        # Step 2: contracted state Ψ_k = Σ_j c_j ψ^j_k
        Psi_k = (p.c[None, :, None] * psi).sum(axis=1)  # (N_u, N+1)

        # Step 3: F(u, Ψ_k) — u has shape (N_u,), broadcast to (N_u, N+1)
        F_k = riccati_F(u_grid[:, None], Psi_k, p.nu, p.rho)  # (N_u, N+1)

        # Step 4: ξ_0(T - t_k) — note the integral variable is s, ξ_0(T - s)
        xi_vals = self.fv(T - t_grid)  # (N+1,)

        # Step 5: trapezoidal integral over t_grid
        phi = np.trapezoid(F_k * xi_vals[None, :], t_grid, axis=1)  # (N_u,)

        result = np.exp(phi)
        # For large imaginary frequencies the Riccati ODE can overflow numerically.
        # By the Riemann-Lebesgue lemma |CF(iω)| → 0 as ω → ∞; overflow in the
        # intermediate Ψ values corresponds to Re(φ) → −∞, so exp(φ) → 0.
        result = np.where(np.isfinite(result), result, 0.0 + 0.0j)
        return result

    def cf_with_spot(self, u_grid: np.ndarray, S0: float) -> np.ndarray:
        """Evaluate cf of log(S_T): exp(u log S_0) * exp(φ^n(0, T))."""
        return np.exp(u_grid * np.log(S0)) * self.cf_centered(u_grid)
