"""Extension 1 — maturity-dependent additive split w(t) (NEW WORK).

The base symmetric model partitions ξ0 by a CONSTANT convex weight,
g0^(1)=(1-w)ξ0, g0^(2)=w ξ0. The erratum's structural worry was that a single
scalar share cannot give the slow block more weight at long maturities (where its
skew contribution is supposed to matter) without taking it from the short end too.

This extension makes the split MATURITY-DEPENDENT while keeping every good property
of the additive construction:

    g0^(1)(t) = (1 - w(t)) ξ0(t),   g0^(2)(t) = w(t) ξ0(t),
    w(t) = w_L + (w_S - w_L) e^{-a t},   w_S, w_L ∈ [0,1],  a > 0.

Because w(t) is a convex blend of two points in [0,1] (e^{-a t} ∈ [0,1]), we have
w(t) ∈ [min(w_S,w_L), max(w_S,w_L)] ⊆ [0,1] for ALL t — so:
  * both forcings are ≥ 0 everywhere → the block-1 input is NEVER clipped;
  * E^Q[V_t] = (1-w(t))ξ0(t) + w(t)ξ0(t) = ξ0(t) EXACTLY at every maturity, for
    any (w_S, w_L, a).
The §8.9.7 clipping bug therefore stays structurally gone, and the split itself now
carries term-structure information: w_S is the short-maturity block-2 share, w_L the
long-maturity share, a the transition rate. The slow block can earn a genuinely
larger share at the long end (w_L > w_S) without starving the short end.

Calibration vector (9 parameters):
    (w_S, w_L, a, H1, nu1, rho1, kappa2, nu2, rho2)
i.e. the 7-parameter base model plus 2 (the constant w becomes the triple w_S,w_L,a).

Implemented entirely by reusing the unmodified single-block lifted-Heston CF for
both blocks, exactly like the base model — only the forcing is now a weighted curve.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
import numpy as np

from ..lifted_heston.params import geometric_grid
from ..lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from ..common.forward_variance import ForwardVariance
from ..common.cos_method import cos_call_prices, cos_truncation_interval
from ..common.black_scholes import bs_implied_vol


def ts_weight(t, w_S: float, w_L: float, a: float):
    """w(t) = w_L + (w_S - w_L) e^{-a t}. Convex blend of w_S, w_L ⇒ stays in [0,1]."""
    t = np.asarray(t, dtype=float)
    return w_L + (w_S - w_L) * np.exp(-a * t)


class WeightedForwardVariance(ForwardVariance):
    """ξ0_weighted(t) = weight(t) · base(t), weight a callable t → [0,1]. No floor."""

    def __init__(self, base: ForwardVariance, weight):
        self.base = base
        self.weight = weight  # callable(t) -> array/scalar in [0,1]

    def __call__(self, t):
        return self.weight(t) * self.base(t)

    def integrated(self, T: float) -> float:
        from scipy.integrate import quad
        val, _ = quad(lambda s: float(self(s)), 0.0, T, limit=200)
        return float(val)


@dataclass
class SymmetricTwoFactorTSParams:
    """9-parameter term-structure-split symmetric two-factor lifted Heston."""

    w_S:    float
    w_L:    float
    a:      float
    H1:     float
    nu1:    float
    rho1:   float
    kappa2: float
    nu2:    float
    rho2:   float
    n1:     int = 20
    r_n1:   float = 2.5
    n2:     int = 1
    H2:     float | None = None
    r_n2:   float = 2.5

    c1: np.ndarray = field(init=False, repr=False)
    x1: np.ndarray = field(init=False, repr=False)
    c2: np.ndarray = field(init=False, repr=False)
    x2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for nm, v in (("w_S", self.w_S), ("w_L", self.w_L)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{nm} must be in [0, 1], got {v}")
        if self.a <= 0:
            raise ValueError(f"a must be > 0, got {self.a}")
        self.c1, self.x1 = geometric_grid(self.H1, self.n1, self.r_n1)
        if not (-1.0 <= self.rho1 <= 1.0) or not (-1.0 <= self.rho2 <= 1.0):
            raise ValueError("rho1, rho2 must be in [-1, 1]")
        if self.nu1 <= 0 or self.nu2 <= 0:
            raise ValueError("nu1, nu2 must be > 0")
        if self.n2 == 1:
            if self.kappa2 <= 0:
                raise ValueError(f"kappa2 must be > 0, got {self.kappa2}")
            self.c2 = np.array([1.0]); self.x2 = np.array([float(self.kappa2)])
        else:
            if self.H2 is None:
                raise ValueError("n2 > 1 requires H2")
            self.c2, self.x2 = geometric_grid(self.H2, self.n2, self.r_n2)

    def weight(self, t):
        return ts_weight(t, self.w_S, self.w_L, self.a)


class SymmetricTwoFactorTSCF:
    """Centered CF of the term-structure-split symmetric two-factor model at fixed T."""

    def __init__(self, params: SymmetricTwoFactorTSParams,
                 market_xi0: ForwardVariance, T: float, n_steps: int = 200):
        self.params = params
        self.market_xi0 = market_xi0
        self.T = float(T)
        self.n_steps = n_steps
        fv1 = WeightedForwardVariance(market_xi0, lambda t: 1.0 - params.weight(t))
        fv2 = WeightedForwardVariance(market_xi0, params.weight)
        proxy1 = types.SimpleNamespace(c=params.c1, x=params.x1, nu=params.nu1, rho=params.rho1)
        proxy2 = types.SimpleNamespace(c=params.c2, x=params.x2, nu=params.nu2, rho=params.rho2)
        self._cf1 = LiftedHestonCharacteristicFunction(proxy1, fv1, self.T, n_steps)
        self._cf2 = LiftedHestonCharacteristicFunction(proxy2, fv2, self.T, n_steps)

    def cf_centered(self, u_grid: np.ndarray) -> np.ndarray:
        return self._cf1.cf_centered(u_grid) * self._cf2.cf_centered(u_grid)


def ts_call_prices(params, market_xi0, S0, strikes, T, N_cos=256, L0=12.0, n_steps=200):
    cf = SymmetricTwoFactorTSCF(params, market_xi0, T, n_steps)
    total_var = market_xi0.integrated(T)   # block1+block2 forcings sum back to ξ0
    a, b = cos_truncation_interval(T, total_var, L0)
    px = cos_call_prices(cf.cf_centered, S0, np.asarray(strikes, float), T, a, b, N_cos)
    return np.maximum(px, 0.0)


def ts_iv_surface(params, market_xi0, S0, strikes_per_T, N_cos=256, L0=12.0, n_steps=200):
    iv = {}
    for T_m, K_arr in strikes_per_T.items():
        K_arr = np.asarray(K_arr, dtype=float)
        px = ts_call_prices(params, market_xi0, S0, K_arr, T_m, N_cos, L0, n_steps)
        iv[float(T_m)] = np.array([
            bs_implied_vol(float(p), S0, float(K), T_m, "C") for p, K in zip(px, K_arr)])
    return iv
