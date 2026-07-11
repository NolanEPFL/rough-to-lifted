"""Two-factor lifted Heston parameters (Chapter 8 of the thesis, Case I).

Definition 8.1. Parameter bundle for

    dS_t/S_t   = √V^(1)_t dB^(1)_t + √V^(2)_t dB^(2)_t
    V^(1)_t    = ξ_0^(1)(t) + Σ c_i U^{e1,i}_t
    dU^{e1,i}_t = (-x_i U^{e1,i}_t) dt + ν_1 √V^(1)_t dW^1_t   (lam1=0, FV form)
    dV^(2)_t   = λ_2 (θ_2 - V^(2)_t) dt + ν_2 √V^(2)_t dW^2_t

with W_1 ⊥ W_2 (Case I, ρ_{12} = 0).

Block 1 uses the same geometric Lévy grid as LiftedHestonParams.
Block 2 is a classical CIR factor parametrised by (λ_2, θ_2, ν_2, ρ_2, V^(2)_0).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..lifted_heston.params import geometric_grid


@dataclass
class TwoFactorLiftedHestonParams:
    """Flat parameter bundle for the two-factor lifted Heston (Case I).

    Attributes
    ----------
    H1, n1, r_n1 : Hurst index, factor count, geometric ratio for block 1.
    nu1, rho1 : vol-of-vol and price-vol correlation for block 1.
    lam2, theta2, nu2, rho2, V2_0 : CIR parameters for block 2.
    c, x : geometric-grid weights/speeds for block 1 (computed, not init args).
    """
    H1:     float
    n1:     int
    r_n1:   float
    nu1:    float
    rho1:   float
    lam2:   float
    theta2: float
    nu2:    float
    rho2:   float
    V2_0:   float

    c: np.ndarray = field(init=False, repr=False)
    x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # geometric_grid validates H1 ∈ (0,0.5), n1 even, r_n1 > 1
        self.c, self.x = geometric_grid(self.H1, self.n1, self.r_n1)
        if not (-1.0 <= self.rho1 <= 1.0):
            raise ValueError(f"rho1 must be in [-1, 1], got {self.rho1}")
        if self.nu1 <= 0:
            raise ValueError(f"nu1 must be > 0, got {self.nu1}")
        if self.lam2 < 0:
            raise ValueError(f"lam2 must be ≥ 0, got {self.lam2}")
        if self.theta2 < 0:
            raise ValueError(f"theta2 must be ≥ 0, got {self.theta2}")
        if self.nu2 < 0:
            raise ValueError(f"nu2 must be ≥ 0, got {self.nu2}")
        if not (-1.0 <= self.rho2 <= 1.0):
            raise ValueError(f"rho2 must be in [-1, 1], got {self.rho2}")
        if self.V2_0 < 0:
            raise ValueError(f"V2_0 must be ≥ 0, got {self.V2_0}")

    def block2_mean(self, t) -> np.ndarray:
        """E_Q[V^(2)_t] = θ_2 + (V^(2)_0 - θ_2) exp(-λ_2 t)."""
        t = np.asarray(t, dtype=float)
        return self.theta2 + (self.V2_0 - self.theta2) * np.exp(-self.lam2 * t)
