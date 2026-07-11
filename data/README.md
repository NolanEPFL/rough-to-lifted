# `data/` directory

This directory holds the option-data inputs used by the experiments.

**No proprietary market data is shipped with this repository.** The SPX/CBOE
option chains used in the thesis come from CBOE DataShop and Polygon.io, whose
licenses do not permit public redistribution. You must obtain your own data and
place it here. The `.gitignore` deliberately excludes `data/*.csv` so that
market data is never committed by accident.

## What is shipped

- `sample_synthetic_spx.csv` — a small, fully synthetic option chain generated
  by `scripts/make_sample_data.py` from an n-factor lifted Heston model. It has
  no licensing restrictions and lets every real-data script run out of the box.
  Regenerate it any time with:

  ```bash
  python scripts/make_sample_data.py
  ```

## Expected CSV format (real data)

`src/data/spx_loader.py::load_spx_csv` reads a **raw single-date option chain**
with exactly these columns (one row per quoted option):

| column           | type   | description                                        |
|------------------|--------|----------------------------------------------------|
| `expiry_date`    | string | option expiry, ISO `YYYY-MM-DD`                    |
| `strike`         | float  | strike (index points)                              |
| `bid`            | float  | bid price                                          |
| `ask`            | float  | ask price                                          |
| `option_type`    | string | `C` or `P`                                         |
| `quote_date`     | string | snapshot date, ISO `YYYY-MM-DD`                    |
| `spot`           | float  | index spot on the snapshot date                    |
| `risk_free_rate` | float  | continuously-compounded risk-free rate             |

Suggested filename: `data/spx_<YYYY-MM-DD>.csv`.

### Cleaning baked into the loader (see `docs/MODEL_SPEC.md`, section 9.2)

1. Mid-price `= (bid + ask) / 2`.
2. Keep `bid > 0`, `ask > 0`, spread `< 50%` of mid.
3. Maturity window `T` in `[7/365, 2.0]` years.
4. Forward per maturity via put-call parity (median over near-ATM pairs).
5. Out-of-the-money only (calls `K > F`, puts `K < F`).
6. Maturity-dependent log-moneyness brackets (Abi Jaber and Li 2025, Table 2).
7. Implied vol by Brent inversion; drop `iv` outside `[0.01, 1.5]`.

### Forward-variance curve

`fit_xi0_from_surface` builds the initial forward-variance curve xi0(t) from the
ATM total-variance term structure (a PCHIP spline differentiated in T). Both the
lifted Heston and aBergomi models are fed the **same** xi0 curve so the
comparison is fair.

## Out-of-sample split

The temporal out-of-sample experiments split by maturity:

- **Calibration (in-sample):** maturities `T <= 1.0` year.
- **Test (out-of-sample):** maturities `T > 1.0` year.

The threshold is `OOS_T_THRESHOLD` in `spx_loader.py`; the loader returns an
`oos_mask` alongside the surface.

## Where to get real SPX option data

- **CBOE DataShop** — official end-of-day SPX chains (`UnderlyingOptionsEODCalcs`).
- **Polygon.io** — options snapshots (`scripts/download_spx_polygon.py --api-key ...`).
- **OptionMetrics IvyDB / WRDS** — via a university subscription.

For a single headline SPX date, one clean end-of-day CBOE snapshot is enough.
