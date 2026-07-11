"""Diagnostic: does the TF-LH deep-OTM put-wing wobble disappear with larger N_cos?

Loads cached TF-LH parameters, reprices the affected slices with N_cos in
[256, 512, 1024, 2048], and produces a 3-row × 3-column figure showing the
full slice, a zoom on the wobble region, and the residual vs the 2048 baseline.

Usage
-----
    python scripts/_diag_ncos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import two_factor_lh_iv_surface

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CSV  = _ROOT / "data" / "spx_2024-12-04.csv"
JSON_PATH = (_ROOT / "results/07b_temporal_oos_two_factor"
             / "2024-12-04_2024-12-05_42_lam0.0/temporal_oos_results.json")
OUT_DIR   = _ROOT / "results/_diag_ncos"

T_DIAG_TARGETS = [0.293, 0.619, 1.537]
N_COS_VALUES   = [256, 512, 1024, 2048]
L0, N_STEPS    = 12.0, 200

VERDICT_THRESHOLD_BPS = 50   # > this → COS is the cause
TARGET_BPS            = 10   # smallest N_cos where residual drops below this


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load surface ──────────────────────────────────────────────────────────
    print(f"Loading {DATA_CSV.name} ...")
    surf  = load_spx_csv(DATA_CSV)
    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv    = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0    = float(surf.forwards.mean())

    K_per_T = surf.strikes_per_T()
    F_per_T = surf.forward_per_T()
    iv_mkt  = surf.ivs_per_T()
    all_T   = np.array(sorted(K_per_T.keys()))

    # ── Load TF-LH params ─────────────────────────────────────────────────────
    with open(JSON_PATH) as f:
        data = json.load(f)
    p = data["results"]["TF-LH"]["params"]
    params = TwoFactorLiftedHestonParams(
        H1=p["H1"], n1=20, r_n1=2.5,
        nu1=p["nu1"], rho1=p["rho1"],
        lam2=p["lam2"], theta2=p["theta2"],
        nu2=p["nu2"], rho2=p["rho2"],
        V2_0=p["V2_0"],
    )

    # ── Pick closest T in surface for each diagnostic target ─────────────────
    T_diag = [float(all_T[np.argmin(np.abs(all_T - t))]) for t in T_DIAG_TARGETS]
    print(f"Diagnostic maturities: {[f'{t:.4f}' for t in T_diag]}")

    # ── Reprice each slice for each N_cos ─────────────────────────────────────
    # results[T][N_cos] = iv array (aligned with K_per_T[T])
    results: dict[float, dict[int, np.ndarray]] = {T: {} for T in T_diag}
    wall_times: dict[float, dict[int, float]] = {T: {} for T in T_diag}

    for T in T_diag:
        K_arr = K_per_T[T]
        strikes_single = {T: K_arr}
        for N in N_COS_VALUES:
            t0 = time.perf_counter()
            iv_dict = two_factor_lh_iv_surface(
                params, fv, S0, strikes_single, N_cos=N, L0=L0, n_steps=N_STEPS
            )
            elapsed = time.perf_counter() - t0
            results[T][N] = iv_dict[T]
            wall_times[T][N] = elapsed
            print(f"  T={T:.4f}  N_cos={N:4d}  ({elapsed*1000:.0f} ms)")

    # ── Colour map for N_cos ──────────────────────────────────────────────────
    cmap   = cm.viridis
    colors = {N: cmap(i / (len(N_COS_VALUES) - 1)) for i, N in enumerate(N_COS_VALUES)}

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("TF-LH N_cos diagnostic — 2024-12-04 IS surface", fontsize=13)

    max_resid: dict[float, dict[int, float]] = {}   # T -> N -> max |resid| bps
    verdict_triggered = False

    for row, T in enumerate(T_diag):
        K_arr  = K_per_T[T]
        F      = F_per_T[T]
        k_arr  = np.log(K_arr / F)
        order  = np.argsort(k_arr)
        k_s    = k_arr[order]
        iv_m_s = iv_mkt[T][order]

        # Baseline = highest-resolution
        N_base  = N_COS_VALUES[-1]
        iv_base = results[T][N_base][order]

        # Wobble / zoom region: leftmost quarter of the smile
        k_zoom_max = -0.2
        k_zoom_min = k_s[0]

        max_resid[T] = {}

        ax_full, ax_zoom, ax_resid = axes[row]

        # ── Left: full slice ──────────────────────────────────────────────────
        fin_mkt = np.isfinite(iv_m_s) & (iv_m_s > 0.01) & (iv_m_s < 1.5)
        ax_full.scatter(k_s[fin_mkt], iv_m_s[fin_mkt],
                        s=18, color="black", zorder=5, label="Market")
        for N in N_COS_VALUES:
            iv_n = results[T][N][order]
            fin  = np.isfinite(iv_n) & (iv_n > 0.01) & (iv_n < 1.5)
            ax_full.plot(k_s[fin], iv_n[fin], color=colors[N],
                         lw=1.2, label=f"N={N}")
        ax_full.set_title(f"T = {T:.4f}y — full slice", fontsize=9)
        ax_full.set_xlabel("log(K/F)"); ax_full.set_ylabel("IV")
        ax_full.legend(fontsize=7); ax_full.grid(True, alpha=0.3)

        # ── Middle: zoom on put wing ──────────────────────────────────────────
        zoom_mask = k_s <= k_zoom_max
        ax_zoom.scatter(k_s[zoom_mask & fin_mkt], iv_m_s[zoom_mask & fin_mkt],
                        s=18, color="black", zorder=5, label="Market")
        for N in N_COS_VALUES:
            iv_n = results[T][N][order]
            fin  = zoom_mask & np.isfinite(iv_n) & (iv_n > 0.01) & (iv_n < 1.5)
            ax_zoom.plot(k_s[fin], iv_n[fin], color=colors[N],
                         lw=1.2, label=f"N={N}")
        ax_zoom.set_xlim(k_zoom_min, k_zoom_max)
        ax_zoom.set_title(f"T = {T:.4f}y — zoom put wing", fontsize=9)
        ax_zoom.set_xlabel("log(K/F)"); ax_zoom.set_ylabel("IV")
        ax_zoom.grid(True, alpha=0.3)

        # ── Right: residual vs 2048 baseline ──────────────────────────────────
        resid_mask = zoom_mask & np.isfinite(iv_base) & (iv_base > 0.01)
        for N in N_COS_VALUES[:-1]:   # skip the baseline itself
            iv_n   = results[T][N][order]
            resid  = (iv_n - iv_base) * 10000   # bps
            fin_r  = resid_mask & np.isfinite(iv_n)
            ax_resid.plot(k_s[fin_r], resid[fin_r], color=colors[N],
                          lw=1.2, label=f"N={N}")
            # max residual on k < -0.2
            wing = fin_r & (k_s < -0.2)
            max_resid[T][N] = float(np.max(np.abs(resid[wing]))) if wing.any() else 0.0
        ax_resid.axhline(0, color="black", lw=0.5, ls="--")
        ax_resid.set_xlim(k_zoom_min, k_zoom_max)
        ax_resid.set_title(f"T = {T:.4f}y — residual vs N=2048 (bps)", fontsize=9)
        ax_resid.set_xlabel("log(K/F)"); ax_resid.set_ylabel("IV diff (bps)")
        ax_resid.legend(fontsize=7); ax_resid.grid(True, alpha=0.3)

    out_path = OUT_DIR / "ncos_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")

    # ── Numerical summary ─────────────────────────────────────────────────────
    print()
    for T in T_diag:
        print(f"T = {T:.4f}y")
        for N in N_COS_VALUES[:-1]:
            r = max_resid[T].get(N, 0.0)
            print(f"  |IV(N={N:4d}) - IV(N=2048)|_max on k<-0.2:  {r:.1f} bps")
            if r > VERDICT_THRESHOLD_BPS:
                verdict_triggered = True
        print(f"  wall time (N=256):  {wall_times[T][256]*1000:.0f} ms")
        print(f"  wall time (N=2048): {wall_times[T][2048]*1000:.0f} ms")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    if verdict_triggered:
        # Find smallest N_cos where all max residuals drop below TARGET_BPS
        sufficient_N = None
        for N in N_COS_VALUES[:-1]:
            if all(max_resid[T].get(N, 0.0) <= TARGET_BPS for T in T_diag):
                sufficient_N = N
                break
        if sufficient_N is None:
            sufficient_N = N_COS_VALUES[-1]
        print(f"VERDICT: COS resolution is the cause. "
              f"Bump N_cos to the smallest value where the max residual drops "
              f"below 10 bps (N_cos = {sufficient_N}).")
    else:
        print("VERDICT: N_cos is NOT the dominant cause. "
              "Investigate the Riccati ODE n_steps next.")


if __name__ == "__main__":
    main()
