"""scripts/cboe_2018_short_end_diagnostic.py - investigate sub-1m skew flattening.

For each sub-month maturity on 2018-06-20 (T < ~0.09 y), produce:

  1. Quote-quality at the 98/102 ATM strikes used by the skew estimator:
     - the bracketing strikes (the two nearest the 0.98 S0 and 1.02 S0 targets)
       in the cleaned surface, their bid/ask/mid and spread-as-pct-of-mid;
     - the number of cleaned-surface strikes within |k_spot| < 0.1 (smile density);
     - flags for: 98/102 spread > 20% of mid, OR the bracket boundary lies further
       than 1% of S0 from the target (suggesting a poor finite difference).

  2. Smoothness / robustness:
     - per-maturity skew, neighbour gaps, sorted monotonicity check;
     - power-law refit dropping flagged-low-quality maturities;
     - power-law refit on a liquid-only set across the FULL maturity range
       (any maturity where the 98/102 bracket has spread > 20% of mid is dropped).

  3. CBOE-IV cross-check on the short end:
     - same 0.98/1.02 spot-moneyness skew, but interpolating CBOE's reported
       implied_volatility_1545 instead of my Brent-inverted IV. Same strikes used.
     - If CBOE IVs give the SAME flattening, the feature is in the market data.

Standalone; does not modify shared code or touch Rough.tex.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.spx_loader import load_spx_csv


SURFACE_CSV = Path(_ROOT) / "data" / "spx_2018-06-20.csv"
RAW_CBOE = Path(_ROOT) / "data" / "cboe" / "UnderlyingOptionsEODCalcs_2018-06-20.csv"
OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
DIAG_JSON = OUT_DIR / "short_end_diagnostic.json"
PER_MAT_CSV = OUT_DIR / "short_end_diagnostic_per_maturity.csv"

S0 = 2769.59
QUOTE_DATE = "2018-06-20"
SHORT_T_CUTOFF_DAYS = 33                 # captures the 11 sub-month maturities (7..30 days)
SPREAD_PCT_THRESHOLD = 20.0              # % of mid; above => flagged low-quality
BRACKET_DIST_PCT_S0 = 1.0                # % of S0; bracket strike further than this from target => flagged
K_LOWER_TARGET = 0.98 * S0
K_UPPER_TARGET = 1.02 * S0
LOG_098 = float(np.log(0.98))
LOG_102 = float(np.log(1.02))


def find_bracket(k_sorted: np.ndarray, K_sorted: np.ndarray, iv_sorted: np.ndarray,
                 k_target: float) -> dict | None:
    """Return the two cleaned-surface strikes bracketing k_target."""
    if k_target < k_sorted[0] or k_target > k_sorted[-1]:
        return None
    j = int(np.searchsorted(k_sorted, k_target))
    if j == 0:
        j = 1
    if j >= len(k_sorted):
        j = len(k_sorted) - 1
    return {
        "K_below": float(K_sorted[j - 1]),
        "K_above": float(K_sorted[j]),
        "iv_below": float(iv_sorted[j - 1]),
        "iv_above": float(iv_sorted[j]),
        "k_below": float(k_sorted[j - 1]),
        "k_above": float(k_sorted[j]),
        "iv_at_target": float(np.interp(k_target, k_sorted, iv_sorted)),
    }


def lookup_raw_otm(raw: pd.DataFrame, expiry_str: str, K: float, F_T: float) -> dict | None:
    """For the OTM-side option (K<F_T => put, K>F_T => call), return bid/ask/spread/cboe_iv."""
    otype = "P" if K < F_T else "C"
    mask = ((raw["expiration"] == expiry_str)
            & (np.isclose(raw["strike"].astype(float), K))
            & (raw["option_type"].astype(str).str.upper() == otype))
    if not mask.any():
        # try the OTHER type as fallback
        otype_alt = "C" if otype == "P" else "P"
        mask = ((raw["expiration"] == expiry_str)
                & (np.isclose(raw["strike"].astype(float), K))
                & (raw["option_type"].astype(str).str.upper() == otype_alt))
        if not mask.any():
            return None
        otype = otype_alt
    row = raw[mask].iloc[0]
    bid = float(row["bid_1545"])
    ask = float(row["ask_1545"])
    mid = (bid + ask) / 2.0
    spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else float("inf")
    return {
        "option_type": otype, "bid": bid, "ask": ask, "mid": mid,
        "spread_pct_of_mid": float(spread_pct),
        "cboe_iv": float(row["implied_volatility_1545"]),
    }


def cboe_iv_skew_at_maturity(raw: pd.DataFrame, expiry_str: str, F_T: float) -> dict | None:
    """Compute the 98/102 skew at this maturity using CBOE's IVs (NOT my Brent inversion).

    OTM-only (put for K<F, call for K>F), CBOE IV > 0. Returns dict with skew and the
    number of points used for the interpolation on each side.
    """
    grp = raw[raw["expiration"] == expiry_str].copy()
    if grp.empty:
        return None
    grp["bid"] = grp["bid_1545"].astype(float)
    grp["ask"] = grp["ask_1545"].astype(float)
    grp["mid"] = (grp["bid"] + grp["ask"]) / 2.0
    grp["cboe_iv"] = grp["implied_volatility_1545"].astype(float)
    grp["otype"] = grp["option_type"].astype(str).str.upper()
    grp["strike_f"] = grp["strike"].astype(float)

    # OTM filter
    is_otm = ((grp["strike_f"] < F_T) & (grp["otype"] == "P")) | \
             ((grp["strike_f"] > F_T) & (grp["otype"] == "C"))
    grp = grp[is_otm & (grp["cboe_iv"] > 0) & (grp["bid"] > 0) & (grp["ask"] > 0)].copy()
    if len(grp) < 4:
        return None

    grp = grp.sort_values("strike_f")
    K = grp["strike_f"].values
    iv = grp["cboe_iv"].values
    k_spot = np.log(K / S0)

    if LOG_098 < k_spot[0] or LOG_102 > k_spot[-1]:
        return None
    iv_down = float(np.interp(LOG_098, k_spot, iv))
    iv_up = float(np.interp(LOG_102, k_spot, iv))
    return {
        "skew_cboe_spot": (iv_down - iv_up) / 0.04,
        "iv_at_098_S0_cboe": iv_down, "iv_at_102_S0_cboe": iv_up,
        "n_otm_points_used": int(len(grp)),
    }


def power_law_fit(T: np.ndarray, s: np.ndarray) -> dict | None:
    mask = np.isfinite(T) & np.isfinite(s) & (s > 0) & (T > 0)
    T_f, s_f = T[mask], s[mask]
    if len(T_f) < 3:
        return None
    x = np.log(T_f); y = np.log(s_f)
    alpha, log_C = np.polyfit(x, y, 1)
    y_hat = log_C + alpha * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "C": float(np.exp(log_C)), "alpha": float(alpha),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(len(T_f)),
        "T_min": float(T_f.min()), "T_max": float(T_f.max()),
    }


def main() -> None:
    print(f"[short-end-diag] loading SPXSurface from {SURFACE_CSV}")
    surf = load_spx_csv(SURFACE_CSV, weight_scheme="equal")

    print(f"[short-end-diag] loading raw CBOE CSV from {RAW_CBOE}")
    raw = pd.read_csv(RAW_CBOE)
    raw["expiration"] = raw["expiration"].astype(str)

    Ks_per_T = surf.strikes_per_T()
    ivs_per_T = surf.ivs_per_T()
    Fs_per_T = surf.forward_per_T()

    quote_dt = pd.Timestamp(QUOTE_DATE)

    short_rows: list[dict] = []
    all_rows: list[dict] = []

    for T in sorted(Ks_per_T.keys()):
        days = int(round(T * 365))
        expiry_dt = quote_dt + pd.Timedelta(days=days)
        expiry_str = expiry_dt.strftime("%Y-%m-%d")

        K = Ks_per_T[T]
        iv = ivs_per_T[T]
        F_T = float(Fs_per_T[T])

        order = np.argsort(K)
        K_sorted = K[order]; iv_sorted = iv[order]
        k_spot_sorted = np.log(K_sorted / S0)

        # Bracketing strikes for the two interpolation targets.
        br_low = find_bracket(k_spot_sorted, K_sorted, iv_sorted, LOG_098)
        br_up = find_bracket(k_spot_sorted, K_sorted, iv_sorted, LOG_102)

        # Smile density near ATM (|k_spot| < 0.1).
        atm_density = int(np.sum(np.abs(k_spot_sorted) < 0.1))

        # Quote quality on the four bracket strikes via the raw CBOE join.
        quote_low_below = lookup_raw_otm(raw, expiry_str, br_low["K_below"], F_T) if br_low else None
        quote_low_above = lookup_raw_otm(raw, expiry_str, br_low["K_above"], F_T) if br_low else None
        quote_up_below = lookup_raw_otm(raw, expiry_str, br_up["K_below"], F_T) if br_up else None
        quote_up_above = lookup_raw_otm(raw, expiry_str, br_up["K_above"], F_T) if br_up else None

        spreads = [q["spread_pct_of_mid"] for q in
                   (quote_low_below, quote_low_above, quote_up_below, quote_up_above)
                   if q is not None]
        max_spread = max(spreads) if spreads else float("nan")

        # Distance from bracket boundary to target (% of S0).
        def _dist_pct(br: dict | None, K_target: float) -> float:
            if br is None: return float("nan")
            return min(abs(br["K_below"] - K_target), abs(br["K_above"] - K_target)) / S0 * 100.0
        dist_low = _dist_pct(br_low, K_LOWER_TARGET)
        dist_up = _dist_pct(br_up, K_UPPER_TARGET)
        max_bracket_dist = max(d for d in (dist_low, dist_up) if not np.isnan(d))

        flag_spread = (max_spread > SPREAD_PCT_THRESHOLD)
        flag_distance = (max_bracket_dist > BRACKET_DIST_PCT_S0)
        flag_low_quality = bool(flag_spread or flag_distance)

        # My (cleaned-surface) skew, spot-moneyness.
        skew_spot = ((br_low["iv_at_target"] - br_up["iv_at_target"]) / 0.04
                     if (br_low is not None and br_up is not None) else None)

        # CBOE-IV skew at this maturity (independent IV path).
        cboe = cboe_iv_skew_at_maturity(raw, expiry_str, F_T)

        row = {
            "T": float(T), "days": days, "expiry": expiry_str, "F_T": F_T,
            "n_strikes_total": int(len(K)),
            "n_strikes_abs_k_lt_0.1": atm_density,
            "K_lower_target_098_S0": float(K_LOWER_TARGET),
            "K_upper_target_102_S0": float(K_UPPER_TARGET),
            "bracket_low_K_below": br_low["K_below"] if br_low else None,
            "bracket_low_K_above": br_low["K_above"] if br_low else None,
            "bracket_up_K_below": br_up["K_below"] if br_up else None,
            "bracket_up_K_above": br_up["K_above"] if br_up else None,
            "quote_low_below": quote_low_below,
            "quote_low_above": quote_low_above,
            "quote_up_below": quote_up_below,
            "quote_up_above": quote_up_above,
            "max_spread_pct_of_mid": float(max_spread),
            "max_bracket_dist_pct_S0": float(max_bracket_dist),
            "flag_spread_above_20pct": bool(flag_spread),
            "flag_bracket_dist_above_1pct_S0": bool(flag_distance),
            "flag_low_quality": flag_low_quality,
            "my_skew_spot": (None if skew_spot is None else float(skew_spot)),
            "cboe_skew_spot": (cboe["skew_cboe_spot"] if cboe else None),
            "cboe_iv_at_098_S0": (cboe["iv_at_098_S0_cboe"] if cboe else None),
            "cboe_iv_at_102_S0": (cboe["iv_at_102_S0_cboe"] if cboe else None),
            "cboe_n_otm_points_used": (cboe["n_otm_points_used"] if cboe else None),
        }
        all_rows.append(row)
        if days <= SHORT_T_CUTOFF_DAYS:
            short_rows.append(row)

    # Monotonicity / smoothness across the short-end maturities.
    short_T = np.array([r["T"] for r in short_rows])
    short_s = np.array([r["my_skew_spot"] if r["my_skew_spot"] is not None else np.nan
                        for r in short_rows])
    order_T = np.argsort(short_T)
    short_T_o = short_T[order_T]; short_s_o = short_s[order_T]
    diffs = np.diff(short_s_o)
    n_up = int(np.sum(diffs > 0)); n_dn = int(np.sum(diffs < 0))
    max_jump = float(np.max(np.abs(diffs)))
    mean_jump = float(np.mean(np.abs(diffs)))

    # Re-fits.
    all_T = np.array([r["T"] for r in all_rows])
    all_s = np.array([r["my_skew_spot"] if r["my_skew_spot"] is not None else np.nan
                      for r in all_rows])
    flag_lq = np.array([r["flag_low_quality"] for r in all_rows])

    fits: dict = {}
    fits["full_range_all"] = power_law_fit(all_T, all_s)
    fits["full_range_drop_flagged"] = power_law_fit(all_T[~flag_lq], all_s[~flag_lq])
    short_mask = all_T < SHORT_T_CUTOFF_DAYS / 365
    fits["short_end_all"] = power_law_fit(all_T[short_mask], all_s[short_mask])
    fits["short_end_drop_flagged"] = power_law_fit(
        all_T[short_mask & ~flag_lq], all_s[short_mask & ~flag_lq]
    )

    summary = {
        "spot": S0,
        "K_lower_target_098_S0": float(K_LOWER_TARGET),
        "K_upper_target_102_S0": float(K_UPPER_TARGET),
        "short_T_cutoff_days": SHORT_T_CUTOFF_DAYS,
        "spread_pct_threshold": SPREAD_PCT_THRESHOLD,
        "bracket_dist_pct_S0_threshold": BRACKET_DIST_PCT_S0,
        "short_end_n_maturities": len(short_rows),
        "short_end_n_flagged_low_quality": int(sum(r["flag_low_quality"] for r in short_rows)),
        "short_end_smoothness": {
            "neighbour_diffs_sign_count_up": n_up,
            "neighbour_diffs_sign_count_down": n_dn,
            "max_abs_neighbour_jump": max_jump,
            "mean_abs_neighbour_jump": mean_jump,
            "monotone_decreasing": bool(n_up == 0 and n_dn > 0),
        },
        "fits": fits,
        "paper_reference": {"C": 0.35, "alpha": -0.41},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[short-end-diag] wrote summary JSON: {DIAG_JSON}")

    # Flatten the per-maturity records (drop the nested quote dicts) for the CSV.
    def _flatten(r: dict) -> dict:
        out = {k: v for k, v in r.items() if k not in
               ("quote_low_below", "quote_low_above", "quote_up_below", "quote_up_above")}
        for tag in ("low_below", "low_above", "up_below", "up_above"):
            q = r.get(f"quote_{tag}")
            if q is None:
                out[f"{tag}_bid"] = None; out[f"{tag}_ask"] = None
                out[f"{tag}_mid"] = None; out[f"{tag}_spread_pct"] = None
                out[f"{tag}_cboe_iv"] = None; out[f"{tag}_otype"] = None
                continue
            out[f"{tag}_bid"] = q["bid"]; out[f"{tag}_ask"] = q["ask"]
            out[f"{tag}_mid"] = q["mid"]; out[f"{tag}_spread_pct"] = q["spread_pct_of_mid"]
            out[f"{tag}_cboe_iv"] = q["cboe_iv"]; out[f"{tag}_otype"] = q["option_type"]
        return out

    pd.DataFrame([_flatten(r) for r in all_rows]).to_csv(PER_MAT_CSV, index=False)
    print(f"[short-end-diag] wrote per-maturity CSV: {PER_MAT_CSV}")

    # Console summary
    print()
    print("=== SHORT-END QUOTE QUALITY (T <= 33 days) ===")
    print(f"{'days':>4} {'F_T':>7} {'#smile':>6} {'max_spread%':>11} {'br_dist%S0':>10} "
          f"{'my_skew':>7} {'cboe_skew':>9} {'flagged':>7}")
    for r in short_rows:
        print(f"{r['days']:>4d} {r['F_T']:>7.2f} {r['n_strikes_abs_k_lt_0.1']:>6d} "
              f"{r['max_spread_pct_of_mid']:>11.2f} {r['max_bracket_dist_pct_S0']:>10.3f} "
              f"{r['my_skew_spot']:>7.4f} "
              f"{(r['cboe_skew_spot'] if r['cboe_skew_spot'] is not None else float('nan')):>9.4f} "
              f"{'YES' if r['flag_low_quality'] else '-':>7}")

    print()
    print("=== SMOOTHNESS (short-end my_skew across consecutive maturities, sorted by T) ===")
    print(f"  neighbour diff signs: {n_up} up, {n_dn} down (monotone_decreasing="
          f"{summary['short_end_smoothness']['monotone_decreasing']})")
    print(f"  max |jump| = {max_jump:.4f}, mean |jump| = {mean_jump:.4f}")

    print()
    print("=== POWER-LAW REFITS ===")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
