"""Two-factor lifted Heston characteristic function.

Theorem 8.7 of the thesis:

    Phi_T(u) = Phi_T^(1)(u) * Phi_T^(2)(u)

Block 1 (rough): the standard lifted Heston CF of block 1 evaluated with
  ξ_0^(1)(t) = ξ_0^market(t) - E_Q[V^(2)_t]

  where E_Q[V^(2)_t] = theta2 + (V2_0 - theta2) * exp(-lam2 * t).
  This ensures the model is consistent with the market forward variance curve.
  (PROMPT_TWO_FACTOR.md §5, last bullet.)

Block 2 (slow): the scalar CIR CF from cir_riccati.py.

The two blocks are independent (rho_12 = 0, Case I), so the joint CF
factorises as a pointwise product of two (N_u,) complex arrays.
"""

from __future__ import annotations

import types
import numpy as np

from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.cir_riccati import cir_cf_centered
from src.lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from src.common.forward_variance import ForwardVariance, FlatForwardVariance


def block1_forward_variance(
    market_xi0: ForwardVariance,
    params: TwoFactorLiftedHestonParams,
) -> ForwardVariance:
    """Build ξ_0^(1)(t) = ξ_0^market(t) - E_Q[V^(2)_t].

    E_Q[V^(2)_t] = theta2 + (V2_0 - theta2) * exp(-lam2 * t).

    We return a ForwardVariance wrapper that subtracts the block-2 mean.
    Clips to a minimum of 1e-8 to prevent negative values for pathological
    inputs (documented in PROMPT_TWO_FACTOR.md §5).
    """
    class _Block1FV(ForwardVariance):
        def __call__(self, t):
            xi_market = market_xi0(t)
            xi_block2 = params.block2_mean(t)
            return np.maximum(xi_market - xi_block2, 1e-8)

        def integrated(self, T: float) -> float:
            from scipy.integrate import quad
            val, _ = quad(self, 0.0, T, limit=200)
            return float(val)

    return _Block1FV()


class TwoFactorLiftedHestonCF:
    """Characteristic function of the two-factor lifted Heston model.

    Parameters
    ----------
    params  : TwoFactorLiftedHestonParams
    market_xi0 : forward variance curve from the market (full ξ_0^market).
    T       : maturity in years.
    n_steps : Riccati time steps (same for both blocks).
    """

    def __init__(
        self,
        params: TwoFactorLiftedHestonParams,
        market_xi0: ForwardVariance,
        T: float,
        n_steps: int = 200,
    ):
        self.params = params
        self.T = float(T)
        self.n_steps = n_steps

        # Build block-1 forward variance (market minus block-2 mean)
        self.xi0_block1 = block1_forward_variance(market_xi0, params)

        # LiftedHestonCharacteristicFunction only accesses p.c, p.x, p.nu, p.rho.
        # Use a SimpleNamespace proxy to avoid importing LiftedHestonParams
        # (cross-package relative import causes pytest collection issues).
        lh_proxy = types.SimpleNamespace(
            c=params.c, x=params.x,
            nu=params.nu1, rho=params.rho1,
        )
        self._lh_cf = LiftedHestonCharacteristicFunction(
            lh_proxy, self.xi0_block1, T, n_steps
        )

    def cf_centered(self, u_grid: np.ndarray) -> np.ndarray:
        """Phi_T(u) = Phi_T^(1)(u) * Phi_T^(2)(u).  Shape: (N_u,) complex."""
        p = self.params
        phi1 = self._lh_cf.cf_centered(u_grid)           # (N_u,)
        phi2 = cir_cf_centered(
            u_grid, p.nu2, p.rho2, p.lam2, p.theta2, p.V2_0,
            self.T, self.n_steps,
        )                                                  # (N_u,)
        return phi1 * phi2

    def cf_with_spot(self, u_grid: np.ndarray, S0: float) -> np.ndarray:
        """exp(u * log S_0) * Phi_T(u)."""
        return np.exp(u_grid * np.log(S0)) * self.cf_centered(u_grid)
