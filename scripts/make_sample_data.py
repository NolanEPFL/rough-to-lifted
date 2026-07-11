"""Generate a small SYNTHETIC option chain in the raw-quote CSV format.

This lets the real-data scripts (04_spx_comparison.py, 07_temporal_oos.py, ...)
run out of the box, without any proprietary market data.  The surface is priced
from an n-factor lifted Heston model with SPX-like parameters, then converted to
Black-76 call/put prices with a tight synthetic bid/ask spread.

Output columns match src/data/spx_loader.py::load_spx_csv exactly:
    expiry_date, strike, bid, ask, option_type, quote_date, spot, risk_free_rate

Usage (from the repository root):
    python scripts/make_sample_data.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.common.black_scholes import bs_call_price, bs_put_price
from src.common.forward_variance import FlatForwardVariance
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface

# --- configuration (deterministic) ------------------------------------------
S0          = 100.0                 # spot; zero rates => forward F = S0
RATE        = 0.0
QUOTE_DATE  = "2024-01-02"
MATURITIES  = [0.083, 0.25, 0.5, 1.0, 1.5, 2.0]
LOG_K_GRID  = np.linspace(-0.40, 0.25, 17)
HALF_SPREAD = 0.005                 # 0.5% half-spread on each option price
OUT_PATH    = os.path.join(_ROOT, "data", "sample_synthetic_spx.csv")

# Ground-truth model: lifted Heston, H = 0.10, moderate vol-of-vol, leverage.
PARAMS = LiftedHestonParams(H=0.10, n=200, r_n=2.5, nu=0.4, rho=-0.7)


def main() -> None:
    strikes = S0 * np.exp(LOG_K_GRID)
    strikes_per_T = {T: strikes for T in MATURITIES}

    fv = FlatForwardVariance(0.04)
    iv_surface = lifted_heston_iv_surface(PARAMS, fv, S0, strikes_per_T)

    quote_dt = datetime.fromisoformat(QUOTE_DATE)
    rows = []
    for T in MATURITIES:
        expiry_dt = quote_dt + timedelta(days=int(round(T * 365)))
        ivs = np.asarray(iv_surface[T], dtype=float)
        for K, iv in zip(strikes, ivs):
            if not np.isfinite(iv) or iv <= 0.0:
                continue
            call = float(bs_call_price(S0, K, T, iv))
            put = float(bs_put_price(S0, K, T, iv))
            for otype, price in (("C", call), ("P", put)):
                if price <= 1e-6:
                    continue
                rows.append({
                    "expiry_date":     expiry_dt.date().isoformat(),
                    "strike":          round(float(K), 4),
                    "bid":             round(price * (1.0 - HALF_SPREAD), 6),
                    "ask":             round(price * (1.0 + HALF_SPREAD), 6),
                    "option_type":     otype,
                    "quote_date":      QUOTE_DATE,
                    "spot":            S0,
                    "risk_free_rate":  RATE,
                })

    df = pd.DataFrame(rows, columns=[
        "expiry_date", "strike", "bid", "ask",
        "option_type", "quote_date", "spot", "risk_free_rate",
    ])
    df.to_csv(OUT_PATH, index=False)
    print(f"[make_sample_data] wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
