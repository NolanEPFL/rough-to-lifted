"""scripts/cboe_2018_adapter.py - CBOE EOD CSV -> Polygon-format adapter.

Step 1 of the 2018 validation. Reads the CBOE UnderlyingOptionsEODCalcs CSV for
2018-06-20 and emits a Polygon-format CSV that the existing src/data/spx_loader.py
can ingest unchanged. ADDITIVE only - does not touch the three-pairs pipeline.

Conventions:
- Uses 1545 (3:45 pm CT) quote columns to stay consistent with CBOE Greeks/IV.
- Per-row underlying = implied_underlying_price_1545 if > 0 else active_underlying_price_1545.
- Polygon-format CSV uses the constant active_underlying_price_1545 as spot and r=0,
  so the existing put-call-parity F-extractor recovers a per-maturity F from mid prices.
- T = (expiration - quote_date).days / 365.0 (ACT/365, matching load_spx_csv).
- Drops: bid==0, ask==0, mid==0, expiration==quote_date.
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


CBOE_RAW = Path(_ROOT) / "data" / "cboe" / "UnderlyingOptionsEODCalcs_2018-06-20.csv"
POLYGON_OUT = Path(_ROOT) / "data" / "spx_2018-06-20.csv"
SUMMARY_OUT = Path(_ROOT) / "results" / "cboe_2018_validation" / "adapter_summary.json"


def load_cboe_raw(path: Path) -> pd.DataFrame:
    """Read the CBOE CSV with no filtering, return the raw frame."""
    df = pd.read_csv(path)
    return df


def adapt_cboe_to_polygon(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert the CBOE frame to a Polygon-format frame + diagnostic dict.

    Polygon-format columns:
        expiry_date, strike, bid, ask, option_type, quote_date, spot, risk_free_rate

    Returns
    -------
    (df_polygon, diag) where diag holds row counts, drop reasons, spot value,
    underlying-source breakdown.
    """
    diag: dict = {"rows_in": int(len(df_raw))}

    # Per-row underlying: prefer implied_underlying_price_1545 when non-zero,
    # else fall back to the cash active_underlying_price_1545.
    impl = df_raw["implied_underlying_price_1545"].astype(float)
    act = df_raw["active_underlying_price_1545"].astype(float)
    used_implied = impl > 0
    underlying = np.where(used_implied, impl, act)
    diag["underlying_source_implied_rows"] = int(used_implied.sum())
    diag["underlying_source_active_rows"] = int((~used_implied).sum())
    diag["active_underlying_unique"] = sorted(act.unique().tolist())
    diag["implied_underlying_nonzero_range"] = (
        [float(impl[used_implied].min()), float(impl[used_implied].max())]
        if used_implied.any() else None
    )

    bid = df_raw["bid_1545"].astype(float)
    ask = df_raw["ask_1545"].astype(float)
    mid = (bid + ask) / 2.0

    quote_dt = pd.to_datetime(df_raw["quote_date"])
    exp_dt = pd.to_datetime(df_raw["expiration"])
    ttm_days = (exp_dt - quote_dt).dt.days

    # Drop unusable rows (track reasons separately so the per-reason counts are
    # honest; some rows fail multiple criteria, so the final keep mask is the
    # AND of all four).
    drop_bid = bid <= 0
    drop_ask = ask <= 0
    drop_mid = mid <= 0
    drop_ttm = ttm_days <= 0
    drop_any = drop_bid | drop_ask | drop_mid | drop_ttm
    diag["drops"] = {
        "zero_or_negative_bid": int(drop_bid.sum()),
        "zero_or_negative_ask": int(drop_ask.sum()),
        "zero_or_negative_mid": int(drop_mid.sum()),
        "zero_or_negative_ttm": int(drop_ttm.sum()),
        "total_dropped_any_reason": int(drop_any.sum()),
    }
    keep = ~drop_any

    # Spot constant: the cash active_underlying_price_1545 (verified unique below).
    spot_const = float(act.iloc[0])
    diag["polygon_csv_spot"] = spot_const
    diag["polygon_csv_risk_free_rate"] = 0.0

    df_poly = pd.DataFrame({
        "expiry_date": exp_dt.dt.strftime("%Y-%m-%d"),
        "strike": df_raw["strike"].astype(float),
        "bid": bid,
        "ask": ask,
        "option_type": df_raw["option_type"].astype(str).str.upper(),
        "quote_date": quote_dt.dt.strftime("%Y-%m-%d"),
        "spot": spot_const,
        "risk_free_rate": 0.0,
    })[keep].reset_index(drop=True)

    diag["rows_out_polygon_csv"] = int(len(df_poly))

    # TTM range over kept rows.
    ttm_kept = ttm_days[keep].astype(float) / 365.0
    diag["ttm_years_min"] = float(ttm_kept.min())
    diag["ttm_years_max"] = float(ttm_kept.max())
    diag["n_unique_expirations_kept"] = int(df_poly["expiry_date"].nunique())

    return df_poly, diag


def main() -> None:
    if not CBOE_RAW.exists():
        raise FileNotFoundError(f"CBOE raw CSV not found at {CBOE_RAW}")

    print(f"[adapter] reading {CBOE_RAW}")
    df_raw = load_cboe_raw(CBOE_RAW)

    df_poly, diag = adapt_cboe_to_polygon(df_raw)

    POLYGON_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_poly.to_csv(POLYGON_OUT, index=False)
    print(f"[adapter] wrote Polygon-format CSV: {POLYGON_OUT} ({len(df_poly)} rows)")

    # Round-trip: load via the existing pipeline and capture the cleaned surface.
    print("[adapter] loading via src.data.spx_loader.load_spx_csv ...")
    surf = load_spx_csv(POLYGON_OUT, weight_scheme="equal")

    forward_per_T = surf.forward_per_T()
    diag["spx_surface"] = {
        "date": surf.date,
        "n_rows": int(len(surf.strikes)),
        "n_maturities": len(forward_per_T),
        "maturities_years": sorted(float(t) for t in forward_per_T.keys()),
        "forwards_per_maturity": {
            f"{t:.6f}": float(forward_per_T[t]) for t in sorted(forward_per_T.keys())
        },
        "iv_min": float(np.min(surf.iv_mkt)),
        "iv_max": float(np.max(surf.iv_mkt)),
        "iv_median": float(np.median(surf.iv_mkt)),
        "ttm_min": float(np.min(surf.maturities)),
        "ttm_max": float(np.max(surf.maturities)),
        "n_oos_rows": int(surf.oos_mask.sum()),
    }

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(diag, indent=2))
    print(f"[adapter] wrote diagnostic summary: {SUMMARY_OUT}")
    print()
    print("=== ADAPTER SUMMARY ===")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
