"""Parameters for the symmetric-split two-factor lifted Heston (erratum §8.9.7).

Calibration vector (7 parameters, default single-factor slow block):

    theta_SYM = (w, H1, nu1, rho1, kappa2, nu2, rho2)

    V^(1)_t = (1-w) ξ0(t) + Σ_{i=1}^{n1} c_i^(1) U^(1,i)_t      (rough block)
    V^(2)_t =   w   ξ0(t) + Σ_{j=1}^{n2} c_j^(2) U^(2,j)_t      (slow block)

    dU^(1,i)_t = -x_i^(1) U^(1,i)_t dt + nu1 √V^(1)_t dW1_t,  U_0 = 0
    dU^(2,j)_t = -x_j^(2) U^(2,j)_t dt + nu2 √V^(2)_t dW2_t,  U_0 = 0

    dS_t/S_t = √V^(1)_t dB^(1)_t + √V^(2)_t dB^(2)_t,
    B^(j) = rho_j W_j + √(1-rho_j²) W_j^perp,  (W1,W1perp,W2,W2perp) independent.

Block 1 (rough): geometric Lévy grid (c^(1),x^(1)) from (H1,n1,r_n1), with the
canonical n1=20, r_n1=2.5 of the thesis — the SAME grid as the existing block 1.

Block 2 (slow): by default a SINGLE factor n2=1 with c^(2)=[1], x^(2)=[kappa2], so
its forward-variance kernel is exp(-kappa2 (T-t)). kappa2 is the OU speed that
plays the role the old constant-θ2 reversion λ2 played; it enters the Riccati as
the LINEAR coefficient -x^(2)=-kappa2 (NOT inside the nonlinearity F2). The
constructor also accepts a multi-factor slow block (n2>1 via a Hurst H2 and ratio
r_n2) so a second lifted-Heston block is possible later, but defaults to n2=1 to
keep the fair-run parameter count at 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..lifted_heston.params import geometric_grid


@dataclass
class SymmetricTwoFactorParams:
    """Flat parameter bundle for the symmetric-split two-factor lifted Heston.

    Attributes
    ----------
    w : block-2 variance share, convex-combination weight in [0, 1].
    H1, n1, r_n1 : Hurst index, factor count, geometric ratio for the rough block.
    nu1, rho1 : vol-of-vol and price-vol correlation for the rough block.
    kappa2 : mean-reversion speed of the (default) single-factor slow block.
    nu2, rho2 : vol-of-vol and price-vol correlation for the slow block.
    n2, H2, r_n2 : slow-block kernel control. Default n2=1 => single OU speed
        kappa2. If n2>1, H2 must be given and a geometric grid (c^(2),x^(2)) is
        built from (H2, n2, r_n2); kappa2 is then unused for the kernel.
    c1, x1 : geometric-grid weights/speeds for block 1 (computed).
    c2, x2 : weights/speeds for block 2 (computed).
    """

    w:      float
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
        if not (0.0 <= self.w <= 1.0):
            raise ValueError(f"w must be in [0, 1], got {self.w}")
        # Block 1: geometric_grid validates H1 ∈ (0,0.5), n1 even, r_n1 > 1.
        self.c1, self.x1 = geometric_grid(self.H1, self.n1, self.r_n1)
        if not (-1.0 <= self.rho1 <= 1.0):
            raise ValueError(f"rho1 must be in [-1, 1], got {self.rho1}")
        if self.nu1 <= 0:
            raise ValueError(f"nu1 must be > 0, got {self.nu1}")
        if not (-1.0 <= self.rho2 <= 1.0):
            raise ValueError(f"rho2 must be in [-1, 1], got {self.rho2}")
        if self.nu2 <= 0:
            raise ValueError(f"nu2 must be > 0, got {self.nu2}")

        # Block 2 kernel.
        if self.n2 == 1:
            if self.kappa2 <= 0:
                raise ValueError(f"kappa2 must be > 0, got {self.kappa2}")
            self.c2 = np.array([1.0])
            self.x2 = np.array([float(self.kappa2)])
        else:
            if self.H2 is None:
                raise ValueError("n2 > 1 requires a Hurst index H2 for the slow block")
            self.c2, self.x2 = geometric_grid(self.H2, self.n2, self.r_n2)

    def forcing_weights(self) -> tuple[float, float]:
        """(scale1, scale2) of the additive split: g0^(1)=scale1·ξ0, g0^(2)=scale2·ξ0."""
        return (1.0 - self.w, self.w)
