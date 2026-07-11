"""Symmetric-split two-factor lifted Heston model (erratum §8.9.7 remedy).

NEW WORK BEYOND THE THESIS. The thesis (§8.9.7, §9) marks the symmetric-split
construction as future work; this package implements it in a separate module,
leaving every committed result and every existing `src/` file untouched.

Conceptual change vs `src.two_factor_lifted_heston` (the asymmetric, clipped model)
----------------------------------------------------------------------------------
The current model is asymmetric: block 1 is a lifted-Heston block carrying the
full term structure through the *subtractive* forcing g0^(1)=ξ0-E[V^(2)] (clipped
at 1e-8), and block 2 is a CIR factor reverting to a constant θ2 with V0^(2)=θ2.
Tying V0^(2) to θ2 caps V0^(2) ≤ min_t ξ0 on a downward-sloping curve, which makes
block 2 inert, and pushing θ2 up drives g0^(1) negative and trips the clip.

Here the two blocks are SYMMETRIC: each is a lifted-Heston-type block in
forward-variance form (λ=0) with its own NON-NEGATIVE deterministic forcing, the
two forcings partitioning ξ0 ADDITIVELY by a convex combination,

    V_t = [ (1-w) ξ0(t) + Σ_i c_i^(1) U_t^(1,i) ]   (rough block, geometric grid)
        + [   w   ξ0(t) + Σ_j c_j^(2) U_t^(2,j) ]   (slow block, default n2=1)

    g0^(1) = (1-w) ξ0,  g0^(2) = w ξ0,  g0^(1)+g0^(2) = ξ0 exactly,  w ∈ [0,1].

Because the split is additive, both forcings are ≥ 0 for any w ∈ [0,1]: the
block-1 input is NEVER clipped, and E^Q[V_t] = (1-w)ξ0(t) + w ξ0(t) = ξ0(t)
holds exactly at every maturity, for any w. The positivity cap on block 2
disappears.

Both blocks reuse the UNMODIFIED single-block lifted-Heston Riccati/CF machinery
(`src.lifted_heston`); they differ only in kernel (geometric vs single OU speed
κ2), forcing weight (1-w vs w), and leverage/vol-of-vol (ρ1,ν1 vs ρ2,ν2).

Public API
----------
    SymmetricTwoFactorParams        — 7-param dataclass (w,H1,nu1,rho1,kappa2,nu2,rho2)
    ScaledForwardVariance           — w·ξ0 wrapper (the additive partition)
    SymmetricTwoFactorCF            — factorised centered CF φ1·φ2
    symmetric_call_prices           — COS call prices
    symmetric_iv_surface            — implied-vol surface
    simulate_symmetric, mc_call_prices — Euler–Maruyama validator + positivity probe
"""

from .params import SymmetricTwoFactorParams
from .scaled_forward_variance import ScaledForwardVariance
from .characteristic_function import SymmetricTwoFactorCF
from .pricing import symmetric_call_prices, symmetric_iv_surface
from .monte_carlo import simulate_symmetric, mc_call_prices

__all__ = [
    "SymmetricTwoFactorParams",
    "ScaledForwardVariance",
    "SymmetricTwoFactorCF",
    "symmetric_call_prices",
    "symmetric_iv_surface",
    "simulate_symmetric",
    "mc_call_prices",
]
