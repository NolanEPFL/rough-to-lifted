"""Two-factor lifted Heston model (Chapter 8 of the thesis, Case I: ρ_{12} = 0).

Public API:

    TwoFactorLiftedHestonParams        — flat parameter dataclass (H1,n1,r_n1,nu1,rho1,lam2,...)
    TwoFactorLiftedHestonCF            — factorised CF (block1 × block2)
    block1_forward_variance            — builds ξ^(1) = ξ^market - E[V^(2)]
    two_factor_lh_call_prices          — COS call prices
    two_factor_lh_iv_surface           — implied vol surface
    cir_riccati_F, solve_cir_riccati   — scalar Riccati ODE for block 2
    cir_cf_centered                    — block-2 CF contribution
    simulate_two_factor, mc_call_prices — Monte Carlo validator
"""

from .params import TwoFactorLiftedHestonParams
from .characteristic_function import TwoFactorLiftedHestonCF, block1_forward_variance
from .pricing import two_factor_lh_call_prices, two_factor_lh_iv_surface
from .cir_riccati import cir_riccati_F, solve_cir_riccati, cir_cf_centered
from .monte_carlo import simulate_two_factor, mc_call_prices

__all__ = [
    "TwoFactorLiftedHestonParams",
    "TwoFactorLiftedHestonCF",
    "block1_forward_variance",
    "two_factor_lh_call_prices",
    "two_factor_lh_iv_surface",
    "cir_riccati_F",
    "solve_cir_riccati",
    "cir_cf_centered",
    "simulate_two_factor",
    "mc_call_prices",
]
