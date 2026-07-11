"""Symmetric-split two-factor lifted Heston characteristic function.

Two INDEPENDENT lifted-Heston Riccati systems in forward-variance form
(λ1 = λ2 = 0, W1 ⊥ W2), so the joint centered CF factorises as a pointwise
product of the two block CFs:

    Φ_centered(u) = exp(φ(0,T)) = Φ^(1)_centered(u) · Φ^(2)_centered(u)

with

    dψ^(1,i)/ds = -x_i^(1) ψ^(1,i) + F1(u, Ψ^(1)),  Ψ^(1) = Σ_i c_i^(1) ψ^(1,i)
    dψ^(2,j)/ds = -x_j^(2) ψ^(2,j) + F2(u, Ψ^(2)),  Ψ^(2) = Σ_j c_j^(2) ψ^(2,j)
    F1(u,v)  = 0.5(u²-u) + ρ1 ν1 u v  + 0.5 ν1² v²      (no λ term)
    F2(u,w_) = 0.5(u²-u) + ρ2 ν2 u w_ + 0.5 ν2² w_²     (no λ term)

    φ(0,T) = ∫_0^T [ F1(u,Ψ^(1)(s)) (1-w) ξ0(T-s)
                   + F2(u,Ψ^(2)(s))   w   ξ0(T-s) ] ds.

Each block is built as a single-block `LiftedHestonCharacteristicFunction`
(UNMODIFIED) fed with its own kernel proxy (c,x,nu,rho) and its scaled forcing
ScaledForwardVariance(market_xi0, 1-w) resp. (market_xi0, w). κ2 enters block 2
purely as the linear Riccati coefficient x^(2)=[κ2] (single factor), exactly the
lifted-Heston structure.

Both block CFs are CENTERED (spot-stripped, cf of log(S_T/S_0)); their product is
the centered aggregate CF. The spot u·log(S0) appears once and is supplied by the
COS pricer through the strike phase factor — `cf_centered` never carries it. The
COS truncation uses the FULL ξ0 (= sum of the two scaled curves).
"""

from __future__ import annotations

import types
import numpy as np

from .params import SymmetricTwoFactorParams
from .scaled_forward_variance import ScaledForwardVariance
from ..lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from ..common.forward_variance import ForwardVariance


class SymmetricTwoFactorCF:
    """Centered CF of the symmetric-split two-factor lifted Heston at fixed T.

    Parameters
    ----------
    params : SymmetricTwoFactorParams
    market_xi0 : full market forward-variance curve ξ0.
    T : maturity in years.
    n_steps : Riccati time steps (same for both blocks). 200 for calibration,
        1600 for the final reprice used in plots.
    """

    def __init__(
        self,
        params: SymmetricTwoFactorParams,
        market_xi0: ForwardVariance,
        T: float,
        n_steps: int = 200,
    ):
        self.params = params
        self.market_xi0 = market_xi0
        self.T = float(T)
        self.n_steps = n_steps

        scale1, scale2 = params.forcing_weights()  # (1-w, w)
        fv1 = ScaledForwardVariance(market_xi0, scale1)  # g0^(1) = (1-w) ξ0
        fv2 = ScaledForwardVariance(market_xi0, scale2)  # g0^(2) =   w   ξ0

        # LiftedHestonCharacteristicFunction reads only (c, x, nu, rho) from its
        # params object; drive each block with a SimpleNamespace proxy (the same
        # pattern the existing two-factor model uses for its block 1).
        proxy1 = types.SimpleNamespace(
            c=params.c1, x=params.x1, nu=params.nu1, rho=params.rho1
        )
        proxy2 = types.SimpleNamespace(
            c=params.c2, x=params.x2, nu=params.nu2, rho=params.rho2
        )
        self._cf1 = LiftedHestonCharacteristicFunction(proxy1, fv1, self.T, n_steps)
        self._cf2 = LiftedHestonCharacteristicFunction(proxy2, fv2, self.T, n_steps)

    def cf_centered(self, u_grid: np.ndarray) -> np.ndarray:
        """Φ_T(u) = Φ^(1)_T(u) · Φ^(2)_T(u). Shape (N_u,) complex."""
        phi1 = self._cf1.cf_centered(u_grid)
        phi2 = self._cf2.cf_centered(u_grid)
        return phi1 * phi2

    def block_cfs(self, u_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The two centered block CFs (φ1, φ2) separately (for diagnostics)."""
        return self._cf1.cf_centered(u_grid), self._cf2.cf_centered(u_grid)

    def cf_with_spot(self, u_grid: np.ndarray, S0: float) -> np.ndarray:
        """exp(u·log S0) · Φ_T(u) (the spot applied once; unused by the COS pricer)."""
        return np.exp(u_grid * np.log(S0)) * self.cf_centered(u_grid)
