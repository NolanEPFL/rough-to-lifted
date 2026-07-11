"""scripts/cboe_2018_iv_check.py - per-option BS-IV cross-check vs CBOE.

Step 1 validation. For every option on 2018-06-20 with a usable mid price and a
CBOE-reported IV, invert the mid via src.common.black_scholes.bs_implied_vol
using the SAME per-maturity put-call-parity forward that load_spx_csv stores in
the SPXSurface (via src.data.spx_loader._extract_forward, with r = 0).

The parity forward F(T) already embeds the market net rate (r - q) - it is
computed as F(T) = K + (C - P) on near-ATM pairs - so we treat F as the true
forward and feed it to bs_implied_vol (which assumes r = q = 0 in the forward
measure).

Compared rows: bid>0, ask>0, mid>0, T in the parity-forward set, CBOE IV>0.
Rows with CBOE IV == 0 are excluded (CBOE's own inversion failed, no reference).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.black_scholes import bs_implied_vol
from src.data.spx_loader import _extract_forward


CBOE_RAW = Path(_ROOT) / "data" / "cboe" / "UnderlyingOptionsEODCalcs_2018-06-20.csv"
OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
SCATTER_PNG = OUT_DIR / "iv_my_vs_cboe.png"
DIFF_HIST_PNG = OUT_DIR / "iv_abs_diff_hist.png"
PER_ROW_CSV = OUT_DIR / "iv_cross_check_per_option.csv"
STATS_JSON = OUT_DIR / "iv_cross_check_stats.json"


def build_comparison_frame(path: Path) -> tuple[pd.DataFrame, dict[float, float]]:
    """Read the CBOE CSV, compute per-maturity parity forwards, and produce the
    comparison frame. Returns (df, forwards_by_T)."""
    df_raw = pd.read_csv(path)

    bid = df_raw["bid_1545"].astype(float)
    ask = df_raw["ask_1545"].astype(float)
    mid = (bid + ask) / 2.0
    K = df_raw["strike"].astype(float)
    quote_dt = pd.to_datetime(df_raw["quote_date"])
    exp_dt = pd.to_datetime(df_raw["expiration"])
    T = (exp_dt - quote_dt).dt.days.astype(float) / 365.0
    opt_type = df_raw["option_type"].astype(str).str.upper()
    cboe_iv = df_raw["implied_volatility_1545"].astype(float)
    spot = float(df_raw["active_underlying_price_1545"].astype(float).iloc[0])

    pre = pd.DataFrame({
        "expiry_date": exp_dt.dt.strftime("%Y-%m-%d"),
        "T": T, "strike": K, "option_type": opt_type,
        "bid": bid, "ask": ask, "mid": mid,
        "cboe_iv": cboe_iv,
    })

    # Per-maturity forward via the SAME _extract_forward the SPXSurface uses.
    # r = 0 (the parity forward already embeds r - q via C - P).
    forwards: dict[float, float] = {}
    valid_for_parity = pre[(pre["bid"] > 0) & (pre["ask"] > 0) & (pre["T"] > 0)].copy()
    for T_val, grp in valid_for_parity.groupby("T"):
        forwards[float(T_val)] = _extract_forward(grp, float(T_val), r=0.0, S0_fallback=spot)

    # Eligibility for comparison.
    eligible = pre[
        (pre["bid"] > 0) & (pre["ask"] > 0) & (pre["mid"] > 0)
        & (pre["T"] > 0) & (pre["cboe_iv"] > 0)
        & (pre["T"].isin(forwards.keys()))
    ].copy().reset_index(drop=True)
    eligible["F"] = eligible["T"].map(forwards)

    my_iv = np.array([
        bs_implied_vol(float(p), float(f), float(k), float(t), option_type=ot)
        for p, f, k, t, ot in zip(eligible["mid"], eligible["F"], eligible["strike"],
                                  eligible["T"], eligible["option_type"])
    ])
    eligible["my_iv"] = my_iv

    print(f"[iv-check] CBOE rows in: {len(df_raw)}")
    print(f"[iv-check] eligible for comparison: {len(eligible)} "
          f"(after bid/ask>0, mid>0, T>0, CBOE_iv>0, T in parity-forward set)")
    print(f"[iv-check] my_iv finite: {int(np.isfinite(my_iv).sum())}/{len(my_iv)} "
          f"({int((~np.isfinite(my_iv)).sum())} NaN from bracketing failure)")
    print(f"[iv-check] {len(forwards)} per-maturity parity forwards computed "
          f"(cash spot = {spot}); first/last F: "
          f"F({min(forwards):.3f}y)={forwards[min(forwards)]:.3f}, "
          f"F({max(forwards):.3f}y)={forwards[max(forwards)]:.3f}")

    return eligible, forwards


def compute_stats(df: pd.DataFrame, forwards: dict[float, float]) -> tuple[dict, pd.DataFrame]:
    finite = df.dropna(subset=["my_iv"]).copy()
    finite["abs_diff"] = (finite["my_iv"] - finite["cboe_iv"]).abs()
    finite["signed_diff"] = finite["my_iv"] - finite["cboe_iv"]
    n = int(len(finite))

    stats: dict = {
        "forward_convention": "per-maturity put-call parity (load_spx_csv._extract_forward, r=0)",
        "n_per_maturity_forwards": len(forwards),
        "n_eligible_for_comparison": int(len(df)),
        "n_my_iv_nan": int(df["my_iv"].isna().sum()),
        "n_compared": n,
        "mean_abs_diff": float(finite["abs_diff"].mean()),
        "median_abs_diff": float(finite["abs_diff"].median()),
        "max_abs_diff": float(finite["abs_diff"].max()),
        "p99_abs_diff": float(finite["abs_diff"].quantile(0.99)),
        "p95_abs_diff": float(finite["abs_diff"].quantile(0.95)),
        "mean_signed_diff": float(finite["signed_diff"].mean()),
        "median_signed_diff": float(finite["signed_diff"].median()),
        "pct_within_0.001": float((finite["abs_diff"] <= 0.001).mean() * 100),
        "pct_within_0.005": float((finite["abs_diff"] <= 0.005).mean() * 100),
        "pct_within_0.01": float((finite["abs_diff"] <= 0.01).mean() * 100),
        "pct_within_0.05": float((finite["abs_diff"] <= 0.05).mean() * 100),
    }

    def _bucket_T(t: float) -> str:
        if t < 30 / 365: return "0_under_1m"
        if t < 90 / 365: return "1_1m_to_3m"
        if t < 365 / 365: return "2_3m_to_1y"
        return "3_over_1y"
    finite["mat_bucket"] = finite["T"].apply(_bucket_T)
    by_mat = finite.groupby("mat_bucket").agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        max_abs_diff=("abs_diff", "max"),
        mean_signed_diff=("signed_diff", "mean"),
        median_signed_diff=("signed_diff", "median"),
    ).to_dict(orient="index")
    stats["by_maturity_bucket"] = {k: {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in by_mat.items()}

    finite["log_k"] = np.log(finite["strike"] / finite["F"])
    def _bucket_mny(k: float) -> str:
        if k < -0.5: return "0_deep_OTM_put"
        if k < -0.1: return "1_OTM_put"
        if k < 0.1: return "2_near_ATM"
        if k < 0.5: return "3_OTM_call"
        return "4_deep_OTM_call"
    finite["mny_bucket"] = finite["log_k"].apply(_bucket_mny)
    by_mny = finite.groupby("mny_bucket").agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        max_abs_diff=("abs_diff", "max"),
        mean_signed_diff=("signed_diff", "mean"),
    ).to_dict(orient="index")
    stats["by_moneyness_bucket"] = {k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in by_mny.items()}

    return stats, finite


def make_plots(finite: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(finite["cboe_iv"], finite["my_iv"], s=4, alpha=0.3, color="C0")
    lo = min(finite["cboe_iv"].min(), finite["my_iv"].min())
    hi = max(finite["cboe_iv"].max(), finite["my_iv"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("CBOE reported IV (implied_volatility_1545)")
    ax.set_ylabel("My IV (bs_implied_vol on mid, F = per-maturity parity forward)")
    ax.set_title(f"CBOE 2018-06-20: BS-IV inversion cross-check\n(n = {len(finite)})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=130)
    plt.close(fig)
    print(f"[iv-check] wrote {SCATTER_PNG}")

    fig, ax = plt.subplots(figsize=(7, 4))
    abs_diff = (finite["my_iv"] - finite["cboe_iv"]).abs()
    ax.hist(np.clip(abs_diff, 0, 0.05), bins=80, color="C0", alpha=0.85)
    ax.set_xlabel("|my_IV - cboe_IV| (clipped at 0.05)")
    ax.set_ylabel("count")
    ax.set_title(f"CBOE 2018-06-20: |IV diff| distribution (n = {len(finite)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIFF_HIST_PNG, dpi=130)
    plt.close(fig)
    print(f"[iv-check] wrote {DIFF_HIST_PNG}")


def main() -> None:
    if not CBOE_RAW.exists():
        raise FileNotFoundError(f"CBOE raw CSV not found at {CBOE_RAW}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, forwards = build_comparison_frame(CBOE_RAW)
    stats, finite = compute_stats(df, forwards)

    finite.to_csv(PER_ROW_CSV, index=False)
    print(f"[iv-check] wrote per-option CSV: {PER_ROW_CSV} ({len(finite)} rows)")

    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(f"[iv-check] wrote stats JSON: {STATS_JSON}")

    make_plots(finite)

    print()
    print("=== IV CROSS-CHECK STATS (per-maturity parity forward) ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
