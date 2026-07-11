"""scripts/cboe_2018_atm_skew.py - empirical ATM skew on 2018-06-20 vs Abi Jaber (2019) Figure 1.

Step 2 of the 2018 validation. Uses the cleaned SPXSurface from Step 1 and the
existing 98/102 estimator in src.calibration.loss._atm_skew_98_102.

Convention note
---------------
The existing estimator uses FORWARD-moneyness k = log(K/F_T) and interpolates
IVs at k = log(0.98) and k = log(1.02) (i.e. K = 0.98 F_T and K = 1.02 F_T).
Abi Jaber (2019) Figure 1 uses SPOT-moneyness k = log(K/S0) (caption defines
k = ln(K/S0), and the y-axis label "ATM skew 98/102" refers to the 0.98 S0 and
1.02 S0 strikes). To compare apples-to-apples with the paper, we pass S0 as the
"F_T" argument to the existing function - that flips the interp grid from
forward-moneyness to spot-moneyness without touching shared code. We also
compute the native forward-moneyness skew alongside for reference.

Fit
---
log-log linear regression of log(skew) on log(T):
    log(skew) = log(C) + alpha * log(T)   ->   skew(T) = C * T^alpha
We report two fits:
  - full range: all T with a valid skew.
  - short end:  T < 0.3 y (the power-law regime; the long end may deviate).
Paper's reference: C ~ 0.35, alpha ~ -0.41.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import load_spx_csv
from src.calibration.loss import _atm_skew_98_102


SURFACE_CSV = Path(_ROOT) / "data" / "spx_2018-06-20.csv"
OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
PLOT_PNG = OUT_DIR / "atm_skew_2018-06-20.png"
TABLE_CSV = OUT_DIR / "atm_skew_per_maturity.csv"
STATS_JSON = OUT_DIR / "atm_skew_fit_stats.json"

S0 = 2769.59                       # cash spot for 2018-06-20 (active_underlying_price_1545)
PAPER_C = 0.35
PAPER_ALPHA = -0.41
SHORT_T_CUTOFF = 0.3                # years; "short end" cutoff for the power-law fit


def compute_skew_table(surface) -> list[dict]:
    """For each maturity T, compute the 98/102 ATM skew under both moneyness conventions."""
    Ks = surface.strikes_per_T()
    ivs = surface.ivs_per_T()
    Fs = surface.forward_per_T()

    rows: list[dict] = []
    for T in sorted(Ks.keys()):
        K = Ks[T]
        iv = ivs[T]
        F_T = float(Fs[T])

        skew_fwd = _atm_skew_98_102(F_T, K, iv)
        skew_spot = _atm_skew_98_102(S0, K, iv)

        rows.append({
            "T": float(T),
            "F_T": F_T,
            "n_strikes": int(len(K)),
            "K_min": float(K.min()),
            "K_max": float(K.max()),
            "skew_forward_moneyness": (None if skew_fwd is None else float(skew_fwd)),
            "skew_spot_moneyness": (None if skew_spot is None else float(skew_spot)),
        })
    return rows


def power_law_fit(T_arr: np.ndarray, skew_arr: np.ndarray) -> dict | None:
    """Log-log linear regression: log(skew) = log(C) + alpha * log(T).
    Returns dict with C, alpha, R^2, n. None if fewer than 3 valid points."""
    mask = np.isfinite(T_arr) & np.isfinite(skew_arr) & (skew_arr > 0) & (T_arr > 0)
    T_f = T_arr[mask]
    s_f = skew_arr[mask]
    if len(T_f) < 3:
        return None
    x = np.log(T_f)
    y = np.log(s_f)
    alpha, log_C = np.polyfit(x, y, 1)
    y_hat = log_C + alpha * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "C": float(np.exp(log_C)),
        "alpha": float(alpha),
        "R2": float(r2),
        "n_points": int(len(T_f)),
        "T_min": float(T_f.min()),
        "T_max": float(T_f.max()),
    }


def make_plot(rows: list[dict], fits: dict) -> None:
    T = np.array([r["T"] for r in rows])
    s_spot = np.array([r["skew_spot_moneyness"] if r["skew_spot_moneyness"] is not None else np.nan
                       for r in rows])
    s_fwd = np.array([r["skew_forward_moneyness"] if r["skew_forward_moneyness"] is not None else np.nan
                      for r in rows])

    valid_spot = np.isfinite(s_spot) & (s_spot > 0)
    valid_fwd = np.isfinite(s_fwd) & (s_fwd > 0)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(T[valid_spot], s_spot[valid_spot], s=42, color="#c62828", marker="o",
               label=f"empirical (spot-moneyness, n={valid_spot.sum()})", zorder=4)
    ax.scatter(T[valid_fwd], s_fwd[valid_fwd], s=22, color="#1565c0", marker="x",
               label=f"empirical (forward-moneyness, n={valid_fwd.sum()})", alpha=0.85, zorder=3)

    T_dense = np.linspace(T[valid_spot].min(), T[valid_spot].max(), 400)
    paper = PAPER_C * T_dense ** PAPER_ALPHA
    ax.plot(T_dense, paper, "--", color="black", lw=1.4,
            label=f"Abi Jaber (2019) Fig.1: 0.35 · T^(−0.41)", zorder=2)

    f = fits.get("spot_full")
    if f is not None:
        my = f["C"] * T_dense ** f["alpha"]
        ax.plot(T_dense, my, "-", color="#c62828", lw=1.2,
                label=f"my fit (full, spot): {f['C']:.3f} · T^({f['alpha']:.3f}), R²={f['R2']:.3f}",
                zorder=2)
    f = fits.get("spot_short")
    if f is not None:
        T_sh = np.linspace(T[valid_spot].min(), min(SHORT_T_CUTOFF, T[valid_spot].max()), 200)
        my_sh = f["C"] * T_sh ** f["alpha"]
        ax.plot(T_sh, my_sh, ":", color="#c62828", lw=1.6,
                label=(f"my fit (T<{SHORT_T_CUTOFF}y, spot): {f['C']:.3f} · T^({f['alpha']:.3f}), "
                       f"R²={f['R2']:.3f}, n={f['n_points']}"),
                zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("ATM skew 98/102  =  (σ(0.98·X) − σ(1.02·X)) / 0.04")
    ax.set_title("S&P 500 ATM skew term structure on 2018-06-20\n(reproduction of Abi Jaber 2019, Figure 1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=140)
    plt.close(fig)
    print(f"[atm-skew] wrote plot: {PLOT_PNG}")


def main() -> None:
    if not SURFACE_CSV.exists():
        raise FileNotFoundError(f"Step 1 Polygon-format CSV not found: {SURFACE_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[atm-skew] loading SPXSurface from {SURFACE_CSV}")
    surf = load_spx_csv(SURFACE_CSV, weight_scheme="equal")
    print(f"[atm-skew] {len(surf.maturities)} cleaned-surface rows over "
          f"{len(np.unique(surf.maturities))} unique maturities; "
          f"using spot S0 = {S0}")

    rows = compute_skew_table(surf)

    T_arr = np.array([r["T"] for r in rows])
    s_spot = np.array([r["skew_spot_moneyness"] if r["skew_spot_moneyness"] is not None else np.nan
                       for r in rows])
    s_fwd = np.array([r["skew_forward_moneyness"] if r["skew_forward_moneyness"] is not None else np.nan
                      for r in rows])

    n_spot = int(np.sum(np.isfinite(s_spot) & (s_spot > 0)))
    n_fwd = int(np.sum(np.isfinite(s_fwd) & (s_fwd > 0)))
    print(f"[atm-skew] valid skew points: spot-moneyness {n_spot}/{len(rows)}, "
          f"forward-moneyness {n_fwd}/{len(rows)}")

    fits: dict = {}
    fits["spot_full"] = power_law_fit(T_arr, s_spot)
    fits["fwd_full"] = power_law_fit(T_arr, s_fwd)
    short = T_arr < SHORT_T_CUTOFF
    fits["spot_short"] = power_law_fit(T_arr[short], s_spot[short])
    fits["fwd_short"] = power_law_fit(T_arr[short], s_fwd[short])
    paper_ref = {"C": PAPER_C, "alpha": PAPER_ALPHA, "source": "Abi Jaber (2019), Fig.1"}

    summary = {
        "spot": float(S0),
        "paper_reference": paper_ref,
        "short_T_cutoff_years": SHORT_T_CUTOFF,
        "fits": fits,
        "n_maturities_total": len(rows),
        "n_valid_skew_spot_moneyness": n_spot,
        "n_valid_skew_forward_moneyness": n_fwd,
    }
    STATS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[atm-skew] wrote stats JSON: {STATS_JSON}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(TABLE_CSV, index=False)
    print(f"[atm-skew] wrote per-maturity CSV: {TABLE_CSV}")

    make_plot(rows, fits)

    print()
    print("=== ATM SKEW FIT SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
