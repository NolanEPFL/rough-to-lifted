"""Smoke check for raw SVI smile fitting.

Loads data/spx_2024-12-04.csv, fits SVI per slice, and produces a two-panel
figure saved to results/sanity_svi/svi_check.png:
  - Top panel (6 sub-plots): market IV dots + SVI fitted line per maturity
  - Bottom left: ATM skew |d sigma/dk|_{k=0}| vs T on log-log with power-law fit
  - Bottom right: table of RMSE statistics

Also prints median, P90, and max of rmse_iv across slices.

Usage
-----
    python scripts/_sanity_svi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import load_spx_csv
from src.data.svi import fit_svi_surface


def main() -> None:
    data_path = _ROOT / "data" / "spx_2024-12-04.csv"
    out_dir   = _ROOT / "results" / "sanity_svi"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} ...")
    surf = load_spx_csv(data_path)

    print("Fitting SVI per slice ...")
    fits = fit_svi_surface(surf)
    print(f"  Fitted {len(fits)} slices.")

    # ── RMSE statistics ───────────────────────────────────────────────────────
    rmses = np.array([f.rmse_iv for f in fits.values()])
    med   = float(np.median(rmses))
    p90   = float(np.percentile(rmses, 90))
    worst = float(np.max(rmses))
    print(f"\nRMSE statistics (in-sample, on IV):")
    print(f"  Median : {med:.4f}  ({med*100:.2f} bps)")
    print(f"  P90    : {p90:.4f}  ({p90*100:.2f} bps)")
    print(f"  Max    : {worst:.4f}  ({worst*100:.2f} bps)")

    # ── Pick 6 maturities log-uniformly ──────────────────────────────────────
    all_T = np.array(sorted(fits.keys()))
    targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), 6))
    idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
    T_plot = all_T[sorted(set(idx))]

    # ── ATM skew term structure ───────────────────────────────────────────────
    T_skew = all_T.copy()
    sk_abs = np.array([abs(fits[T].atm_skew_derivative()) for T in T_skew])

    # Power-law fit: |skew| = C * T^(-alpha) for T >= 1/12
    T_MIN_FIT = 1.0 / 12.0
    mask_fit  = T_skew >= T_MIN_FIT
    if mask_fit.sum() >= 2:
        log_T = np.log(T_skew[mask_fit])
        log_sk = np.log(sk_abs[mask_fit])
        coeffs = np.polyfit(log_T, log_sk, 1)
        alpha  = float(-coeffs[0])
        C      = float(np.exp(coeffs[1]))
        print(f"\nPower-law fit |d sigma/dk| = C * T^(-alpha) for T >= 1/12:")
        print(f"  C     = {C:.4f}")
        print(f"  alpha = {alpha:.4f}")
    else:
        alpha, C = None, None

    # ── Build figure ──────────────────────────────────────────────────────────
    n_smile = len(T_plot)
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig.suptitle("SVI sanity check — spx_2024-12-04", fontsize=13)

    # Top row: smile panels
    gs_top = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                               top=0.92, bottom=0.48)
    k_fine = np.linspace(-0.30, 0.20, 200)

    for idx_ax, T in enumerate(T_plot[:6]):
        row, col = divmod(idx_ax, 3)
        ax = fig.add_subplot(gs_top[row, col])
        fit = fits[T]

        # Market dots
        mask = surf.maturities == T
        F  = surf.forwards[mask][0]
        Ks = surf.strikes[mask]
        k_mkt = np.log(Ks / F)
        iv_mkt = surf.iv_mkt[mask]
        order = np.argsort(k_mkt)
        ax.scatter(k_mkt[order], iv_mkt[order],
                   s=15, color="black", zorder=5, label="Market")

        # SVI fit line
        ax.plot(k_fine, fit.iv(k_fine), color="steelblue", lw=1.5, label="SVI")

        ax.set_title(f"T = {T:.3f}y  (RMSE={fit.rmse_iv*100:.1f} bps)", fontsize=8)
        ax.set_xlabel("k = log(K/F)", fontsize=7)
        ax.set_ylabel("IV", fontsize=7)
        ax.set_xlim(-0.32, 0.22)
        ax.grid(True, alpha=0.3)
        if idx_ax == 0:
            ax.legend(fontsize=7)

    # Bottom left: ATM skew log-log
    gs_bot = fig.add_gridspec(1, 2, top=0.40, bottom=0.06)
    ax_sk = fig.add_subplot(gs_bot[0, 0])
    ax_sk.scatter(T_skew, sk_abs, s=20, color="black", zorder=5, label="SVI skew")
    if alpha is not None:
        T_line = np.linspace(T_skew[mask_fit][0], T_skew[-1], 200)
        ax_sk.plot(T_line, C * T_line**(-alpha), color="red", lw=1.5,
                   label=fr"$C \cdot T^{{-\alpha}}$, $\alpha={alpha:.2f}$")
    ax_sk.set_xscale("log"); ax_sk.set_yscale("log")
    ax_sk.set_xlabel("T (years)"); ax_sk.set_ylabel(r"$|\partial\sigma/\partial k|_{k=0}|$")
    ax_sk.set_title("ATM skew term structure")
    ax_sk.legend(fontsize=8); ax_sk.grid(True, alpha=0.3, which="both")

    # Bottom right: RMSE per slice
    ax_rm = fig.add_subplot(gs_bot[0, 1])
    ax_rm.bar(range(len(all_T)), rmses * 100, color="steelblue", alpha=0.7)
    ax_rm.axhline(med * 100, color="red", ls="--", lw=1, label=f"Median {med*100:.1f} bps")
    ax_rm.axhline(0.5, color="orange", ls=":", lw=1, label="50 bps threshold")
    ax_rm.set_xlabel("Slice index (sorted by T)")
    ax_rm.set_ylabel("RMSE (bps)")
    ax_rm.set_title("In-sample RMSE per maturity")
    ax_rm.legend(fontsize=7); ax_rm.grid(axis="y", alpha=0.3)

    out_path = out_dir / "svi_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
