"""Scaled forward-variance wrapper: the additive partition of ξ0.

The symmetric split forces block j with g0^(j)(t) = scale_j · ξ0(t), where
scale_1 = 1-w and scale_2 = w. Both scales are in [0, 1] for w ∈ [0, 1], so each
forcing is non-negative wherever ξ0 ≥ 0 — there is NO clip. The two scales sum to
1, so g0^(1)(t) + g0^(2)(t) = ξ0(t) exactly at every t.

This wrapper plugs straight into the unmodified
`LiftedHestonCharacteristicFunction`, which evaluates the forcing as
ξ0(T - t_grid) on the Riccati grid and reads `integrated(T)` only for callers
that want the COS truncation (we use the FULL ξ0 for that, since the two scaled
curves sum back to ξ0).
"""

from __future__ import annotations

import numpy as np

from ..common.forward_variance import ForwardVariance


class ScaledForwardVariance(ForwardVariance):
    """ξ0_scaled(t) = scale · base(t). No positivity floor is ever applied."""

    def __init__(self, base: ForwardVariance, scale: float):
        if scale < 0.0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        self.base = base
        self.scale = float(scale)

    def __call__(self, t):
        return self.scale * self.base(t)

    def integrated(self, T: float) -> float:
        return self.scale * self.base.integrated(T)
