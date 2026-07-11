"""Forward variance curve ξ_0(t) used as input to lifted Heston (with λ=0) and aBergomi.

We define an abstract callable interface and two concrete realisations:
- FlatForwardVariance(V0): ξ_0(t) ≡ V0. Used for synthetic experiments.
- PiecewiseConstantForwardVariance(maturities, levels): cadlag piecewise constant,
  matching the 2025 Abi Jaber & Li convention. Used for SPX experiments.

The class also exposes the integrated variance ∫_0^T ξ_0(t) dt which is needed for
the COS truncation interval and the Black-Scholes recovery sanity check at η = 0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ForwardVariance(ABC):
    """Abstract forward variance curve."""

    @abstractmethod
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate ξ_0(t)."""

    @abstractmethod
    def integrated(self, T: float) -> float:
        """Compute ∫_0^T ξ_0(t) dt."""


class FlatForwardVariance(ForwardVariance):
    """ξ_0(t) ≡ V0."""

    def __init__(self, V0: float):
        if V0 <= 0:
            raise ValueError(f"V0 must be > 0, got {V0}")
        self.V0 = float(V0)

    def __call__(self, t):
        return np.full_like(np.asarray(t, dtype=float), self.V0) if np.ndim(t) else self.V0

    def integrated(self, T: float) -> float:
        return self.V0 * T


class PiecewiseConstantForwardVariance(ForwardVariance):
    """Piecewise constant cadlag forward variance.

    ξ_0(t) = ξ_i for t ∈ [T_i, T_{i+1}), with T_0 = 0.

    Parameters
    ----------
    knots : array of strictly positive maturities, sorted ascending. T_1, ..., T_N.
    levels : array of positive levels ξ_0, ξ_1, ..., ξ_{N-1} corresponding to
             the intervals [0, T_1), [T_1, T_2), ..., [T_{N-1}, T_N).
    """

    def __init__(self, knots: np.ndarray, levels: np.ndarray):
        knots = np.asarray(knots, dtype=float)
        levels = np.asarray(levels, dtype=float)
        if knots.ndim != 1 or levels.ndim != 1:
            raise ValueError("knots and levels must be 1-d arrays")
        if len(knots) != len(levels):
            raise ValueError("knots and levels must have the same length")
        if np.any(knots <= 0):
            raise ValueError("all knots must be strictly positive")
        if len(knots) > 1 and np.any(np.diff(knots) <= 0):
            raise ValueError("knots must be strictly ascending")
        if np.any(levels <= 0):
            raise ValueError("all levels must be strictly positive")
        self._knots = knots
        self._levels = levels

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        idx = np.searchsorted(self._knots, t, side="right")
        idx = np.clip(idx, 0, len(self._levels) - 1)
        result = self._levels[idx]
        return float(result[0]) if scalar else result

    def integrated(self, T: float) -> float:
        result = 0.0
        t_prev = 0.0
        for i, t_knot in enumerate(self._knots):
            if T <= t_prev:
                break
            t_end = min(T, t_knot)
            result += self._levels[i] * (t_end - t_prev)
            t_prev = t_knot
        if T > t_prev:
            result += self._levels[-1] * (T - t_prev)
        return result
