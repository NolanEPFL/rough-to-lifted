"""Extract SPX/SPXW end-of-day quotes from a Polygon options flatfile.

Reads the full ~90GB quotes_YYYY-MM-DD.csv.gz, filters to SPX/SPXW tickers,
keeps the last quote at or before a target time (default 15:45 ET), then
parses each ticker into (expiry, strike, option_type) and saves a clean CSV
ready for load_spx_csv.

Usage
-----
    python scripts/extract_spx_eod_quotes.py \
        --input "C:/Users/nolan/Downloads/massive_test/quotes_2024-01-19.csv.gz" \
        --date 2024-01-19 \
        --time 15:45

Output
------
    data/spx_2024-01-19.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ── Target timestamp helpers ───────────────────────────────────────────────

def _target_ns(date_str: str, time_str: str) -> int:
    """Convert 'YYYY-MM-DD' + 'HH:MM' (ET) to nanoseconds UTC."""
    ET = timezone(timedelta(hours=-5))   # EST (January = no DST)
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    dt = dt.replace(tzinfo=ET)
    return int(dt.timestamp() * 1_000_000_000)


# ── Ticker parser ──────────────────────────────────────────────────────────

_TICKER_RE = re.compile(
    r"^O:(SPXW?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"
)

def _parse_ticker(ticker: str) -> tuple | None:
    """Parse 'O:SPXW240119C04750000' → (expiry, strike, option_type) or None."""
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    _, yy, mm, dd, cp, strike_raw = m.groups()
    expiry = f"20{yy}-{mm}-{dd}"
    strike = int(strike_raw) / 1000.0
    return expiry, strike, ("C" if cp == "C" else "P")


# ── Main extraction ────────────────────────────────────────────────────────

def extract(input_path: Path, date_str: str, time_str: str,
            out_dir: Path, rfr: float, chunk_size: int = 500_000,
            window_minutes: int = 30) -> Path:

    cutoff_ns = _target_ns(date_str, time_str)
    # Only accept quotes within the last `window_minutes` before cutoff
    window_ns = window_minutes * 60 * 1_000_000_000
    floor_ns  = cutoff_ns - window_ns

    print(f"Accepting quotes between "
          f"{datetime.fromtimestamp(floor_ns/1e9, tz=timezone.utc).strftime('%H:%M')} and "
          f"{datetime.fromtimestamp(cutoff_ns/1e9, tz=timezone.utc).strftime('%H:%M')} UTC "
          f"({window_minutes}-min window before {time_str} ET)", flush=True)

    # best[ticker] = (bid, ask, timestamp_ns)
    best: dict[str, tuple[float, float, int]] = {}

    print(f"Streaming {input_path.name} …", flush=True)
    total_rows = 0
    spx_rows   = 0

    for chunk in pd.read_csv(
        input_path,
        compression="gzip",
        chunksize=chunk_size,
        dtype={
            "ticker":         str,
            "bid_price":      float,
            "ask_price":      float,
            "sip_timestamp":  "Int64",
        },
        usecols=["ticker", "bid_price", "ask_price", "sip_timestamp"],
    ):
        total_rows += len(chunk)

        # Keep only SPX/SPXW quotes within the window [floor_ns, cutoff_ns]
        spx = chunk[
            chunk["ticker"].str.startswith("O:SPX") &
            (chunk["sip_timestamp"] >= floor_ns) &
            (chunk["sip_timestamp"] <= cutoff_ns) &
            (chunk["bid_price"] > 0) &
            (chunk["ask_price"] > 0)
        ]
        spx_rows += len(spx)

        # For each ticker in this chunk, keep the row with the largest timestamp
        for ticker, grp in spx.groupby("ticker", sort=False):
            latest = grp.loc[grp["sip_timestamp"].idxmax()]
            ts  = int(latest["sip_timestamp"])
            bid = float(latest["bid_price"])
            ask = float(latest["ask_price"])
            if ticker not in best or ts > best[ticker][2]:
                best[ticker] = (bid, ask, ts)

        rows_M = total_rows / 1_000_000
        print(f"  {rows_M:.1f}M rows read | {len(best)} SPX contracts so far",
              end="\r", flush=True)

    print(f"\nDone. {total_rows:,} total rows | {spx_rows:,} SPX rows | "
          f"{len(best)} unique contracts kept", flush=True)

    if not best:
        print("ERROR: no SPX quotes found before cutoff.", file=sys.stderr)
        sys.exit(1)

    # ── Build output dataframe ─────────────────────────────────────────────
    print("Parsing tickers and building CSV …", flush=True)
    records = []
    for ticker, (bid, ask, _) in best.items():
        parsed = _parse_ticker(ticker)
        if parsed is None:
            continue
        expiry, strike, option_type = parsed
        records.append({
            "expiry_date":    expiry,
            "strike":         strike,
            "bid":            bid,
            "ask":            ask,
            "option_type":    option_type,
            "quote_date":     date_str,
            "spot":           0.0,     # filled below from Yahoo
            "risk_free_rate": rfr,
        })

    df = pd.DataFrame(records)
    df = df[df["expiry_date"] > date_str]   # remove same-day expiries
    print(f"Rows after expiry filter: {len(df)}", flush=True)

    # ── Spot from Yahoo ────────────────────────────────────────────────────
    print("Fetching SPX spot from Yahoo Finance …", flush=True)
    try:
        import yfinance as yf
        hist = yf.Ticker("^GSPC").history(
            start=date_str,
            end=(datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if not hist.empty:
            spot = float(hist["Close"].iloc[-1])
        else:
            raise RuntimeError("empty history")
        print(f"  SPX spot = {spot:.2f}", flush=True)
    except Exception as e:
        print(f"  WARNING: Yahoo failed ({e}) — inferring spot from put-call parity",
              flush=True)
        spot = _infer_spot(df, rfr)
        print(f"  SPX spot ≈ {spot:.2f}", flush=True)

    df["spot"] = spot

    # ── Save ───────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spx_{date_str}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"\nRun comparison with:")
    print(f"  python scripts/04_spx_comparison.py --data {out_path} "
          f"--quote-date {date_str} --de-maxiter 30 --M-paths 50000 --lh-l2")
    return out_path


def _infer_spot(df: pd.DataFrame, rfr: float) -> float:
    """Fallback: estimate spot from put-call parity on nearest expiry."""
    nearest = df["expiry_date"].min()
    grp = df[df["expiry_date"] == nearest].copy()
    grp["mid"] = (grp["bid"] + grp["ask"]) / 2
    calls = grp[grp["option_type"] == "C"].set_index("strike")["mid"]
    puts  = grp[grp["option_type"] == "P"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if common.empty:
        return 5000.0
    diffs = (calls[common] - puts[common]).abs()
    K_star = diffs.idxmin()
    F = K_star + (calls[K_star] - puts[K_star])
    return float(F)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, type=Path,
                        help="Path to quotes_YYYY-MM-DD.csv.gz flatfile")
    parser.add_argument("--date",   required=True,
                        help="Quote date YYYY-MM-DD")
    parser.add_argument("--time",   default="15:45",
                        help="Cutoff time ET (default 15:45)")
    parser.add_argument("--rfr",    type=float, default=0.053,
                        help="Risk-free rate annualised (default 0.053)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("C:/Users/nolan/Downloads/rough_vol_compare/data"),
                        help="Output directory")
    args = parser.parse_args()

    extract(args.input, args.date, args.time, args.out_dir, args.rfr)


if __name__ == "__main__":
    main()
