"""Download SPX options data from Polygon.io and save a cleaned CSV.

Usage
-----
    python scripts/download_spx_polygon.py --api-key YOUR_KEY --date 2024-01-19
    python scripts/download_spx_polygon.py --api-key YOUR_KEY          # today

Output
------
    data/spx_<date>.csv

Columns (matches src/data/spx_loader.load_spx_csv):
    expiry_date, strike, bid, ask, option_type, quote_date, spot, risk_free_rate

Notes
-----
- SPX (European AM-settled) is used; SPXW (European PM-settled weeklies) included
  via the underlying_ticker filter which covers both.
- The Polygon /v3/snapshot/options endpoint is paginated at 250 rows per request;
  this script follows next_url links until all contracts are fetched.
- For historical dates: Polygon Advanced gives access to the End-of-Day
  options snapshot for any past trading day via the same endpoint with
  the ?date= parameter (requires the Advanced / Stocks Advanced plan).
- risk_free_rate: fetched from FRED (SOFR) if --rfr is not given; falls back to 0.05.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

POLYGON_BASE = "https://api.polygon.io"
# SPX index ticker on Polygon; SPXW weekly options also appear under SPX
UNDERLYING   = "SPX"
MAX_RETRIES  = 5
RETRY_DELAY  = 2.0   # seconds between 429 retries


def _get(session: requests.Session, url: str, params: dict, api_key: str) -> dict:
    params = {"apiKey": api_key, **params}
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=60)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  Network error ({e.__class__.__name__}), retrying in {wait:.0f}s …",
                  flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 429:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  Rate limited, waiting {wait:.0f}s …", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def _fetch_snapshot(session: requests.Session, api_key: str,
                    quote_date: str | None) -> list[dict]:
    """Fetch all SPX option contracts from the Polygon snapshot endpoint."""
    url    = f"{POLYGON_BASE}/v3/snapshot/options/{UNDERLYING}"
    params: dict = {"limit": 250}
    if quote_date:
        params["date"] = quote_date   # historical EOD snapshot

    rows: list[dict] = []
    page  = 0
    while True:
        page += 1
        data = _get(session, url, params, api_key)
        results = data.get("results", [])
        rows.extend(results)
        print(f"  Page {page}: {len(results)} contracts fetched"
              f" (total so far: {len(rows)})", flush=True)

        next_url = data.get("next_url")
        if not next_url:
            break
        # next_url already contains the cursor; switch to it
        url    = next_url
        params = {}   # cursor carries all params

    return rows


def _fetch_snapshot_safe(session: requests.Session, api_key: str,
                         quote_date: str | None,
                         min_dte: int = 7, max_dte: int = 730) -> list[dict]:
    """Like _fetch_snapshot but saves partial results on network failure.

    min_dte / max_dte filter by expiration date so we only fetch options
    with meaningful time to expiry (avoids filling pages with 0DTE contracts).
    """
    from datetime import date, timedelta
    today = date.fromisoformat(quote_date) if quote_date else date.today()
    url    = f"{POLYGON_BASE}/v3/snapshot/options/{UNDERLYING}"
    params: dict = {
        "limit": 250,
        "expiration_date.gte": (today + timedelta(days=min_dte)).isoformat(),
        "expiration_date.lte": (today + timedelta(days=max_dte)).isoformat(),
    }

    rows: list[dict] = []
    page  = 0
    while True:
        page += 1
        try:
            data = _get(session, url, params, api_key)
        except RuntimeError as e:
            print(f"\n  WARNING: stopped at page {page} due to error: {e}", flush=True)
            print(f"  Saving {len(rows)} contracts collected so far …", flush=True)
            break
        results = data.get("results", [])
        rows.extend(results)
        print(f"  Page {page}: {len(results)} contracts fetched"
              f" (total so far: {len(rows)})", flush=True)

        next_url = data.get("next_url")
        if not next_url:
            break
        url    = next_url
        params = {}

    return rows


def _sofr_rate() -> float:
    """Attempt to fetch SOFR from FRED; return 0.05 on failure."""
    try:
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR",
            timeout=10,
        )
        lines = r.text.strip().splitlines()
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip():
                return float(parts[1].strip()) / 100.0
    except Exception:
        pass
    return 0.05


def _fetch_spot_yahoo() -> float:
    """Fetch the latest SPX close from Yahoo Finance (^GSPC)."""
    import yfinance as yf
    spx = yf.Ticker("^GSPC")
    hist = spx.history(period="1d")
    if hist.empty:
        raise RuntimeError("yfinance returned empty data for ^GSPC.")
    return float(hist["Close"].iloc[-1])


def _infer_spot_from_options(results: list[dict], rfr: float) -> float:
    """Estimate SPX spot via put-call parity on the nearest expiry.

    For the shortest expiry, finds the strike minimising |C_mid - P_mid|,
    then applies: F = K + (C - P), S ≈ F (rates negligible for short T).
    """
    from collections import defaultdict
    by_expiry: dict[str, dict] = defaultdict(lambda: {"C": {}, "P": {}})
    for r in results:
        details = r.get("details", {})
        quote   = r.get("last_quote", {})
        expiry  = details.get("expiration_date")
        ctype   = details.get("contract_type", "")
        strike  = details.get("strike_price")
        bid     = quote.get("bid")
        ask     = quote.get("ask")
        if None in (expiry, ctype, strike, bid, ask) or bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        side = "C" if ctype.lower().startswith("c") else "P"
        by_expiry[expiry][side][float(strike)] = mid

    if not by_expiry:
        raise RuntimeError("No valid options to infer spot.")

    nearest = sorted(by_expiry.keys())[0]
    calls = by_expiry[nearest]["C"]
    puts  = by_expiry[nearest]["P"]
    common = set(calls) & set(puts)
    if not common:
        raise RuntimeError("No matched call/put strikes in nearest expiry.")

    best_K = min(common, key=lambda k: abs(calls[k] - puts[k]))
    F = best_K + (calls[best_K] - puts[best_K])
    return float(F)


def _parse_rows(results: list[dict], quote_date: str, spot: float, rfr: float) -> pd.DataFrame:
    """Extract the columns we need from Polygon snapshot results."""
    records = []
    for r in results:
        details = r.get("details", {})
        quote   = r.get("last_quote", {})

        strike  = details.get("strike_price")
        expiry  = details.get("expiration_date")
        ctype   = details.get("contract_type", "")
        bid     = quote.get("bid")
        ask     = quote.get("ask")

        if any(v is None for v in [strike, expiry, ctype, bid, ask]):
            continue
        if bid <= 0 or ask <= 0:
            continue

        option_type = "C" if ctype.lower().startswith("c") else "P"
        records.append({
            "expiry_date":    expiry,
            "strike":         float(strike),
            "bid":            float(bid),
            "ask":            float(ask),
            "option_type":    option_type,
            "quote_date":     quote_date,
            "spot":           spot,
            "risk_free_rate": rfr,
        })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SPX options from Polygon.io")
    parser.add_argument("--api-key", required=True, help="Polygon.io API key")
    parser.add_argument(
        "--date", default=None,
        help="Quote date YYYY-MM-DD (omit for today / most recent trading day)",
    )
    parser.add_argument(
        "--rfr", type=float, default=None,
        help="Risk-free rate (annualised, e.g. 0.053). Default: fetch SOFR from FRED.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data"),
        help="Directory for output CSV (default: data/)",
    )
    args = parser.parse_args()

    # Polygon Advanced plan returns the live snapshot regardless of --date.
    # Always use today as the quote date.
    d = date.today()
    if d.weekday() >= 5:   # Saturday / Sunday → roll back to Friday
        d -= timedelta(days=d.weekday() - 4)
    quote_date = d.isoformat()
    if args.date and args.date != quote_date:
        print(f"  Note: historical --date is not available on this plan; "
              f"using today ({quote_date}) instead.", flush=True)

    # Risk-free rate
    if args.rfr is not None:
        rfr = args.rfr
    else:
        print("Fetching SOFR from FRED …", flush=True)
        rfr = _sofr_rate()
        print(f"  Using rfr = {rfr:.4f}", flush=True)

    session = requests.Session()

    print(f"Downloading SPX options (live snapshot, quote_date={quote_date}) …", flush=True)
    results = _fetch_snapshot_safe(session, args.api_key, quote_date)

    print("Fetching SPX spot from Yahoo Finance …", flush=True)
    try:
        spot = _fetch_spot_yahoo()
        print(f"  SPX spot = {spot:.2f}", flush=True)
    except Exception as e:
        print(f"  Yahoo failed ({e}), inferring from put-call parity …", flush=True)
        try:
            spot = _infer_spot_from_options(results, rfr)
            print(f"  SPX spot ≈ {spot:.2f}", flush=True)
        except RuntimeError as e2:
            print(f"  WARNING: {e2} — falling back to 5500.0", flush=True)
            spot = 5500.0

    print(f"Total contracts received: {len(results)}", flush=True)

    if not results:
        print("ERROR: No data returned. Check your API key and plan level.", file=sys.stderr)
        sys.exit(1)

    df = _parse_rows(results, quote_date, spot, rfr)
    print(f"Valid rows after parsing: {len(df)}", flush=True)

    if df.empty:
        print("ERROR: DataFrame is empty after parsing.", file=sys.stderr)
        sys.exit(1)

    # Remove contracts expiring before or on quote_date
    df = df[df["expiry_date"] > quote_date]
    print(f"Rows after removing expired contracts: {len(df)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"spx_{quote_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")
    print(f"\nRun the comparison with:")
    print(f"  python scripts/04_spx_comparison.py --data {out_path} "
          f"--quote-date {quote_date} --de-maxiter 30 --M-paths 50000 --lh-l2")


if __name__ == "__main__":
    main()
