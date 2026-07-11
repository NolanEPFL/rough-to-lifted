"""Calibration loss: vega-weighted RMSE on IVs + optional 98/102 ATM-skew penalty.

The composite loss is:
    L_total(λ) = L_wIV + λ · L_skew       0 ≤ λ < ∞
    L_total(∞) = L_skew                    pure skew calibration

L_wIV  = vega-weighted IV-RMSE over in-sample (K, T) pairs.
L_skew = RMSE of the 98/102 ATM-skew across maturities:
           skew(T) = (σ(0.98·F_T) − σ(1.02·F_T)) / 0.04
         evaluated via linear interpolation in log-strike space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..common.black_scholes import bs_vega


# ── ATM skew helper ───────────────────────────────────────────────────────────

def _atm_skew_98_102(F_T: float, K: np.ndarray, iv: np.ndarray) -> float | None:
    """ATM skew := (σ(0.98·F_T) − σ(1.02·F_T)) / 0.04
    via linear interpolation in log-strike space.

    Returns None if the strike range does not cover [0.98·F_T, 1.02·F_T].
    The denominator 0.04 ≈ log(1.02) − log(0.98) matches Figure 1 of Abi Jaber 2019.
    """
    finite = np.isfinite(iv)
    K_f, iv_f = K[finite], iv[finite]
    if len(K_f) < 2:
        return None

    k = np.log(K_f / F_T)
    order = np.argsort(k)
    k_s, iv_s = k[order], iv_f[order]

    k_down = np.log(0.98)   # ≈ -0.02020
    k_up   = np.log(1.02)   # ≈ +0.01980

    if k_down < k_s[0] or k_up > k_s[-1]:
        return None

    iv_down = float(np.interp(k_down, k_s, iv_s))
    iv_up   = float(np.interp(k_up,   k_s, iv_s))
    return (iv_down - iv_up) / 0.04


# ── Loss class ────────────────────────────────────────────────────────────────

@dataclass
class IVSurfaceLoss:
    """Encapsulates a market IV surface and the composite loss function.

    Attributes
    ----------
    market_ivs      : dict {T: np.ndarray of IVs}.
    strikes         : dict {T: np.ndarray of absolute strikes}.
    S0              : spot price (used for vega weights only).
    forwards        : dict {T: F_T} forward price per maturity.
                      If not provided, falls back to S0 for all T.
    weights         : 'equal' or 'vega' (default 'equal').
    skew_lambda     : weight on ATM-skew RMSE term.  0 = off, np.inf = pure skew.
    in_sample_T_max : if not None, __call__ uses T ≤ this; out_of_sample uses T > this.
    """
    market_ivs:      dict[float, np.ndarray]
    strikes:         dict[float, np.ndarray]
    S0:              float
    forwards:        dict[float, float]        = field(default_factory=dict)
    weights:         str                       = "equal"
    skew_lambda:     float                     = 0.0
    in_sample_T_max: float | None             = None

    def _fwd(self, T: float) -> float:
        """Forward price for maturity T; falls back to S0."""
        return self.forwards.get(T, self.S0)

    def _weighted_sq(
        self,
        model_ivs: dict[float, np.ndarray],
        mats: list[float],
    ) -> tuple[float, float, list[float]]:
        """Return (sum_weighted_sq, sum_weights, skew_sq_list) over given maturities."""
        sq_total = 0.0
        w_total  = 0.0
        skew_sq: list[float] = []

        for T in mats:
            mkt = self.market_ivs[T]
            mdl = model_ivs.get(T, np.full_like(mkt, np.nan))
            K   = self.strikes[T]

            valid = np.isfinite(mkt) & np.isfinite(mdl)
            if not valid.any():
                continue

            mkt_v, mdl_v, K_v = mkt[valid], mdl[valid], K[valid]
            diff = mdl_v - mkt_v

            if self.weights == "equal":
                w = np.ones(valid.sum())
            elif self.weights == "vega":
                w = np.maximum(bs_vega(self.S0, K_v, T, mkt_v), 1e-10)
            else:
                raise ValueError(f"Unknown weight scheme: '{self.weights}'")

            sq_total += float((diff ** 2 * w).sum())
            w_total  += float(w.sum())

            # Skew penalty — always computed when skew_lambda != 0
            if self.skew_lambda != 0:
                F_T   = self._fwd(T)
                s_mkt = _atm_skew_98_102(F_T, K_v, mkt_v)
                s_mdl = _atm_skew_98_102(F_T, K_v, mdl_v)
                if s_mkt is not None and s_mdl is not None:
                    skew_sq.append((s_mdl - s_mkt) ** 2)

        return sq_total, w_total, skew_sq

    def __call__(self, model_ivs: dict[float, np.ndarray]) -> float:
        """Composite loss (in-sample).

        λ = 0:   pure vega-weighted IV-RMSE
        0<λ<∞:  IV-RMSE + λ · skew-RMSE
        λ = ∞:  pure skew-RMSE
        """
        if self.in_sample_T_max is not None:
            mats = [T for T in self.market_ivs if T <= self.in_sample_T_max]
        else:
            mats = list(self.market_ivs.keys())

        sq_total, w_total, skew_sq = self._weighted_sq(model_ivs, mats)

        if np.isinf(self.skew_lambda):
            if not skew_sq:
                return 1e6
            return float(np.sqrt(np.mean(skew_sq)))

        if w_total < 1e-12:
            return 1e6

        rmse = float(np.sqrt(sq_total / w_total))

        if self.skew_lambda > 0 and skew_sq:
            rmse += self.skew_lambda * float(np.sqrt(np.mean(skew_sq)))

        return rmse

    def out_of_sample(self, model_ivs: dict[float, np.ndarray]) -> float | None:
        """IV-RMSE on T > in_sample_T_max only.  Returns None if no OOS maturities."""
        if self.in_sample_T_max is None:
            return None
        mats = [T for T in self.market_ivs if T > self.in_sample_T_max]
        if not mats:
            return None
        sq_total, w_total, _ = self._weighted_sq(model_ivs, mats)
        if w_total < 1e-12:
            return None
        return float(np.sqrt(sq_total / w_total))

    def components(
        self,
        model_ivs: dict[float, np.ndarray],
        oos: bool = False,
    ) -> dict[str, float]:
        """Diagnostic decomposition: IV-RMSE and skew-RMSE independently.

        Works regardless of skew_lambda (even if 0 or inf).
        Returns {'iv_rmse': float, 'skew_rmse': float}.
        The iv_rmse uses the same weight scheme as __call__.
        The skew_rmse uses the 98/102 convention regardless of skew_lambda.
        """
        if self.in_sample_T_max is not None:
            if oos:
                mats = [T for T in self.market_ivs if T > self.in_sample_T_max]
            else:
                mats = [T for T in self.market_ivs if T <= self.in_sample_T_max]
        else:
            mats = list(self.market_ivs.keys())

        sq_total = 0.0
        w_total  = 0.0
        skew_sq: list[float] = []

        for T in mats:
            mkt = self.market_ivs[T]
            mdl = model_ivs.get(T, np.full_like(mkt, np.nan))
            K   = self.strikes[T]

            valid = np.isfinite(mkt) & np.isfinite(mdl)
            if not valid.any():
                continue

            mkt_v, mdl_v, K_v = mkt[valid], mdl[valid], K[valid]
            diff = mdl_v - mkt_v

            if self.weights == "equal":
                w = np.ones(valid.sum())
            elif self.weights == "vega":
                w = np.maximum(bs_vega(self.S0, K_v, T, mkt_v), 1e-10)
            else:
                raise ValueError(f"Unknown weight scheme: '{self.weights}'")

            sq_total += float((diff ** 2 * w).sum())
            w_total  += float(w.sum())

            F_T   = self._fwd(T)
            s_mkt = _atm_skew_98_102(F_T, K_v, mkt_v)
            s_mdl = _atm_skew_98_102(F_T, K_v, mdl_v)
            if s_mkt is not None and s_mdl is not None:
                skew_sq.append((s_mdl - s_mkt) ** 2)

        iv_rmse   = float(np.sqrt(sq_total / w_total)) if w_total > 1e-12 else float("nan")
        skew_rmse = float(np.sqrt(np.mean(skew_sq)))   if skew_sq        else float("nan")
        return {"iv_rmse": iv_rmse, "skew_rmse": skew_rmse}
