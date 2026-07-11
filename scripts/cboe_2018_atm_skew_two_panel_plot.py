"""scripts/cboe_2018_atm_skew_two_panel_plot.py - regenerate the appendix figure.

Reads the COMMITTED per-maturity skew CSV and fit-stats JSON from the
results/ folder (does NOT recompute the 98/102 skew, does NOT recalibrate)
and re-draws the appendix's Figure A.1 as a two-panel layout:

  (a) LINEAR scale, matching Abi Jaber (2019) Fig.1's presentation:
      empirical spot-moneyness dots, my fitted power law, paper reference.
  (b) LOG-LOG scale, the existing presentation: empirical spot + forward
      moneyness, my full + short-end fits, paper reference.

Saves to both
  figures/validation/atm_skew_2018-06-20.png         (consumed by Rough.tex)
  results/cboe_2018_validation/atm_skew_2018-06-20.png  (mirror copy)
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


RES_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
FIG_DIR = Path(_ROOT) / "figures" / "validation"
CSV_IN = RES_DIR / "atm_skew_per_maturity.csv"
JSON_IN = RES_DIR / "atm_skew_fit_stats.json"
OUT_FIG = FIG_DIR / "atm_skew_2018-06-20.png"
OUT_FIG_MIRROR = RES_DIR / "atm_skew_2018-06-20.png"

PAPER_C = 0.35
PAPER_ALPHA = -0.41
SHORT_T_CUTOFF = 0.3   # years; same as the original Step-2 script


def main() -> None:
    df = pd.read_csv(CSV_IN)
    stats = json.loads(JSON_IN.read_text())
    f_full = stats["fits"]["spot_full"]
    f_short = stats["fits"]["spot_short"]
    C_full = float(f_full["C"]); alpha_full = float(f_full["alpha"])
    R2_full = float(f_full["R2"]); n_full = int(f_full["n_points"])
    C_short = float(f_short["C"]); alpha_short = float(f_short["alpha"])
    R2_short = float(f_short["R2"]); n_short = int(f_short["n_points"])

    T = df["T"].values
    s_spot = df["skew_spot_moneyness"].values
    s_fwd = df["skew_forward_moneyness"].values
    valid_spot = np.isfinite(s_spot) & (s_spot > 0)
    valid_fwd = np.isfinite(s_fwd) & (s_fwd > 0)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # --- Panel (a): LINEAR scale (paper presentation) ---
    T_dense_lin = np.linspace(0.005, T[valid_spot].max(), 600)
    ax_lin.scatter(T[valid_spot], s_spot[valid_spot], s=44, color="#c62828",
                   marker="o", label="empirical (spot-moneyness)", zorder=4)
    ax_lin.plot(T_dense_lin, C_full * T_dense_lin ** alpha_full,
                "-", color="#c62828", lw=1.4,
                label=f"my fit  ${C_full:.3f}\\,T^{{{alpha_full:.3f}}}$, $R^2={R2_full:.3f}$",
                zorder=3)
    ax_lin.plot(T_dense_lin, PAPER_C * T_dense_lin ** PAPER_ALPHA,
                "--", color="black", lw=1.3,
                label=r"Abi Jaber (2019) Fig. 1:  $0.35\,T^{-0.41}$", zorder=2)
    y_max = float(np.nanmax([
        np.nanmax(s_spot[valid_spot]),
        np.nanmax(C_full * T_dense_lin ** alpha_full),
        np.nanmax(PAPER_C * T_dense_lin ** PAPER_ALPHA),
    ]))
    ax_lin.set_xlim(0.0, float(T[valid_spot].max()) * 1.02)
    ax_lin.set_ylim(0.0, y_max * 1.05)
    ax_lin.set_xlabel("maturity $T$ (years)")
    ax_lin.set_ylabel("ATM skew 98/102")
    ax_lin.set_title("(a) linear scale (paper's presentation)")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Panel (b): LOG-LOG (the existing presentation) ---
    ax_log.scatter(T[valid_spot], s_spot[valid_spot], s=44, color="#c62828",
                   marker="o", label=f"empirical (spot-moneyness, $n={int(valid_spot.sum())}$)",
                   zorder=4)
    ax_log.scatter(T[valid_fwd], s_fwd[valid_fwd], s=22, color="#1565c0",
                   marker="x", label=f"empirical (forward-moneyness, $n={int(valid_fwd.sum())}$)",
                   alpha=0.85, zorder=3)
    T_dense_log = np.logspace(np.log10(T[valid_spot].min()),
                              np.log10(T[valid_spot].max()), 400)
    ax_log.plot(T_dense_log, C_full * T_dense_log ** alpha_full,
                "-", color="#c62828", lw=1.2,
                label=(f"my fit (full):  ${C_full:.3f}\\,T^{{{alpha_full:.3f}}}$, "
                       f"$R^2={R2_full:.3f}$"), zorder=3)
    T_short = np.logspace(np.log10(T[valid_spot].min()),
                          np.log10(min(SHORT_T_CUTOFF, T[valid_spot].max())), 200)
    ax_log.plot(T_short, C_short * T_short ** alpha_short,
                ":", color="#c62828", lw=1.6,
                label=(f"my fit ($T<{SHORT_T_CUTOFF}\\,$y):  "
                       f"${C_short:.3f}\\,T^{{{alpha_short:.3f}}}$, $n={n_short}$"),
                zorder=3)
    ax_log.plot(T_dense_log, PAPER_C * T_dense_log ** PAPER_ALPHA,
                "--", color="black", lw=1.3,
                label=r"Abi Jaber (2019) Fig. 1:  $0.35\,T^{-0.41}$", zorder=2)
    ax_log.set_xscale("log"); ax_log.set_yscale("log")
    ax_log.set_xlabel("maturity $T$ (years)")
    ax_log.set_ylabel("ATM skew 98/102")
    ax_log.set_title("(b) log-log scale")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.suptitle("S&P 500 ATM skew term structure on 2018-06-20 "
                 "(cf. Abi Jaber 2019, Figure 1)", fontsize=12.5, y=1.00)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    fig.savefig(OUT_FIG_MIRROR, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[two-panel] wrote {OUT_FIG}")
    print(f"[two-panel] wrote {OUT_FIG_MIRROR} (mirror)")
    print(f"[two-panel] {int(valid_spot.sum())} empirical (spot) + "
          f"{int(valid_fwd.sum())} empirical (forward) maturities re-plotted; "
          f"NO skews recomputed.")


if __name__ == "__main__":
    main()
