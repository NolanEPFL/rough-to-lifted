"""Lifted Heston parametrisation: geometric Lévy-measure grid.

Implements equations (4.74)-(4.75) of the thesis (eq. (3.3) of Abi Jaber 2019).

For n even and r_n > 1, the partition endpoints are η_i^n = r_n^{i - 1 - n/2},
i = 0, 1, ..., n. The weights and speeds are

    α := H + 1/2

    c_i^n = (r_n^{1-α} - 1) r_n^{(α-1)(1+n/2)}      (1-α) i
            ─────────────────────────────────── × r_n
                  Γ(α) Γ(2 - α)

            (1 - α)   r_n^{2-α} - 1
    x_i^n = ─────── × ─────────────── × r_n^{i - 1 - n/2}
            (2 - α)   r_n^{1-α} - 1

We strictly require c_i, x_i > 0; the formulas give that automatically for
H ∈ (0, 1/2) and r_n > 1, but we add an assertion to be safe.

A LiftedHestonParams dataclass bundles the model parameters (n, H, r_n, ν, ρ)
along with the resulting (c_i, x_i) for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.special import gamma as Gamma_fn


def geometric_grid(H: float, n: int, r_n: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Compute (c_i, x_i) for i = 1, ..., n via the geometric Lévy-measure grid.

    Parameters
    ----------
    H : Hurst index in (0, 1/2). For H ≥ 1/2 the formulas have removable
        singularities; raise ValueError for now (we never calibrate at H = 1/2).
    n : number of factors. Must be even (Abi Jaber's symmetry condition).
    r_n : geometric ratio > 1. Default 2.5 (Abi Jaber's empirical sweet spot
          for n = 20).

    Returns
    -------
    c : 1d array of length n, all entries strictly positive.
    x : 1d array of length n, sorted ascending, all entries strictly positive.

    Raises
    ------
    ValueError if inputs are out of range.
    """
    if not (0.0 < H < 0.5):
        raise ValueError(f"H must be in (0, 0.5), got {H}")
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"n must be a positive even integer, got {n}")
    if r_n <= 1.0:
        raise ValueError(f"r_n must be > 1, got {r_n}")

    alpha = H + 0.5
    i = np.arange(1, n + 1, dtype=float)

    # eq. (4.74): weights c_i
    c = (
        (r_n ** (1.0 - alpha) - 1.0)
        * r_n ** ((alpha - 1.0) * (1.0 + n / 2.0))
        / (Gamma_fn(alpha) * Gamma_fn(2.0 - alpha))
        * r_n ** ((1.0 - alpha) * i)
    )

    # eq. (4.75): speeds x_i
    x = (
        (1.0 - alpha) / (2.0 - alpha)
        * (r_n ** (2.0 - alpha) - 1.0) / (r_n ** (1.0 - alpha) - 1.0)
        * r_n ** (i - 1.0 - n / 2.0)
    )

    assert np.all(c > 0), f"Some c_i ≤ 0 (min={c.min():.3e})"
    assert np.all(x > 0), f"Some x_i ≤ 0 (min={x.min():.3e})"
    assert np.all(np.diff(x) > 0), "x_i not strictly ascending"

    return c, x


@dataclass
class LiftedHestonParams:
    """Bundle of lifted Heston parameters in the forward-variance form (λ = 0).

    Attributes
    ----------
    H : Hurst index.
    n : number of factors.
    r_n : geometric ratio.
    nu : vol-of-vol > 0.
    rho : leverage in [-1, 1].
    c, x : derived weights and speeds.
    """
    H:      float
    n:      int
    r_n:    float
    nu:     float
    rho:    float
    kernel: str   = "geometric"   # 'geometric' (default) or 'l2' (L²-fit kernel)
    c: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.kernel == "geometric":
            self.c, self.x = geometric_grid(self.H, self.n, self.r_n)
        elif self.kernel == "l2":
            from ..abergomi.kernel_fit import abergomi_l2_kernel
            self.c, self.x, _ = abergomi_l2_kernel(self.H, self.n, seed=42)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'; use 'geometric' or 'l2'")
