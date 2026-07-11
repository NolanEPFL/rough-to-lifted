"""Diagnostic: does the TF-LH deep-OTM wobble converge as Riccati ODE n_steps grows?

Earlier diagnostics ruled out:
- COS resolution (_diag_ncos.py): 0 bps residual across N_cos in {256..2048}.
- BS inversion (_diag_price_vs_iv.py): wobble present in raw log-call-prices.

This script tests whether the wobble is Riccati ODE truncation error (fixable by
raising n_steps) or real model output (not a numerical artifact).

Usage
-----
    python scripts/_diag_nsteps.py
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
from matplotlib.colors import Normalize

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import two_factor_lh_call_prices

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CSV  = _ROOT / "data" / "spx_2024-12-04.csv"
JSON_PATH = (_ROOT / "results/07b_temporal_oos_two_factor"
             / "2024-12-04_2024-12-05_42_lam0.0/temporal_oos_results.json")
OUT_DIR   = _ROOT / "results/_diag_nsteps"

T_DIAG_TARGETS = [0.293, 0.619, 1.537]
N_STEPS_VALUES = [200, 400, 800, 1600, 3200]
N_COS, L0      = 256, 12.0
WING_K_MAX     = -0.2   # deep OTM region: k < this

VERDICT_LARGE_RESID  = 1e-3   # > this at n=200 → under-resolved
VERDICT_FLAT_RESID   = 1e-5   # < this at ALL n → already converged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load surface ──────────────────────────────────────────────────────────
    print(f"Loading {DATA_CSV.name} ...")
    surf = load_spx_csv(DATA_CSV)
    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv   = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0   = float(surf.forwards.mean())

    K_per_T = surf.strikes_per_T()
    F_per_T = surf.forward_per_T()
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

    # ── Closest T for each target ─────────────────────────────────────────────
    T_diag = [float(all_T[np.argmin(np.abs(all_T - t))]) for t in T_DIAG_TARGETS]
    print(f"Diagnostic maturities: {[f'{t:.4f}' for t in T_diag]}")

    # ── Reprice: results[T][n_steps] = call price array (sorted by k) ─────────
    results:    dict[float, dict[int, np.ndarray]] = {T: {} for T in T_diag}
    wall_times: dict[float, dict[int, float]]      = {T: {} for T in T_diag}
    k_sorted:   dict[float, np.ndarray]            = {}
    K_sorted:   dict[float, np.ndarray]            = {}

    for T in T_diag:
        K_arr  = K_per_T[T]
        F      = F_per_T[T]
        k_arr  = np.log(K_arr / F)
        order  = np.argsort(k_arr)
        k_sorted[T] = k_arr[order]
        K_sorted[T] = K_arr[order]

        for n in N_STEPS_VALUES:
            t0 = time.perf_counter()
            raw = two_factor_lh_call_prices(
                params, fv, S0, K_sorted[T], T, N_cos=N_COS, L0=L0, n_steps=n
            )
            elapsed = time.perf_counter() - t0
            results[T][n]    = np.maximum(raw, 0.0)
            wall_times[T][n] = elapsed
            print(f"  T={T:.4f}  n_steps={n:4d}  ({elapsed*1000:.0f} ms)")

    # ── Colourmap ─────────────────────────────────────────────────────────────
    cmap   = cm.viridis
    norm   = Normalize(vmin=0, vmax=len(N_STEPS_VALUES) - 1)
    colors = {n: cmap(norm(i)) for i, n in enumerate(N_STEPS_VALUES)}
    N_BASE = N_STEPS_VALUES[-1]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    fig.suptitle(
        "TF-LH n_steps diagnostic — deep OTM put wing (k < -0.2) — 2024-12-04 IS",
        fontsize=13,
    )

    summary: list[dict] = []

    for row, T in enumerate(T_diag):
        k_s  = k_sorted[T]
        wing = k_s < WING_K_MAX
        k_w  = k_s[wing]

        p_base = results[T][N_BASE]
        # log-safe: only where baseline price is positive
        pos_base = wing & (p_base > 0)
        k_pb     = k_s[pos_base]
        log_base = np.log(p_base[pos_base] + 1e-300)

        ax_log, ax_resid, ax_abs = axes[row]

        # ── Col 0: log-call-price vs k for all n_steps ────────────────────────
        for n in N_STEPS_VALUES:
            p_n = results[T][n]
            pos_n = pos_base & (p_n > 0)
            if pos_n.any():
                ax_log.plot(k_s[pos_n], np.log(p_n[pos_n] + 1e-300),
                            color=colors[n], lw=1.5 if n == N_BASE else 1.0,
                            label=f"n={n}", zorder=5 if n == N_BASE else 3)
        ax_log.set_title(f"T={T:.4f}y  log-price vs k", fontsize=9)
        ax_log.set_xlabel("log(K/F)"); ax_log.set_ylabel("log(call price)")
        ax_log.legend(fontsize=7); ax_log.grid(True, alpha=0.3)
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_log, label="n_steps index", shrink=0.7)

        # ── Col 1 & 2: residuals vs n=3200 baseline ───────────────────────────
        row_summary = dict(T=T, resids={})
        for n in N_STEPS_VALUES[:-1]:
            p_n   = results[T][n]
            pos_n = pos_base & (p_n > 0)
            if not pos_n.any():
                row_summary["resids"][n] = 0.0
                continue
            log_n    = np.log(p_n[pos_n]    + 1e-300)
            log_b_n  = np.log(p_base[pos_n] + 1e-300)
            resid    = log_n - log_b_n
            abs_r    = np.abs(resid)
            row_summary["resids"][n] = float(np.max(abs_r))

            # Col 1: signed residual
            ax_resid.plot(k_s[pos_n], resid, color=colors[n], lw=1.2,
                          label=f"n={n}")
            # Col 2: |residual| log y
            valid_abs = abs_r > 0
            if valid_abs.any():
                ax_abs.semilogy(k_s[pos_n][valid_abs], abs_r[valid_abs],
                                color=colors[n], lw=1.2, label=f"n={n}")

        ax_resid.axhline(0, color="black", lw=0.5, ls="--")
        ax_resid.set_title(f"T={T:.4f}y  log-price residual vs n=3200", fontsize=9)
        ax_resid.set_xlabel("log(K/F)"); ax_resid.set_ylabel("log(p_n) - log(p_3200)")
        ax_resid.legend(fontsize=7); ax_resid.grid(True, alpha=0.3)

        ax_abs.set_title(f"T={T:.4f}y  |residual| log-scale", fontsize=9)
        ax_abs.set_xlabel("log(K/F)"); ax_abs.set_ylabel("|log(p_n) - log(p_3200)|")
        ax_abs.legend(fontsize=7); ax_abs.grid(True, alpha=0.3)

        summary.append(row_summary)

    out_path = OUT_DIR / "nsteps_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")

    # ── Numerical summary and verdict ─────────────────────────────────────────
    print()
    all_under_resolved = True
    all_flat           = True
    any_non_monotone   = False

    for s in summary:
        T     = s["T"]
        resids = s["resids"]   # n -> max |log-price resid| on k < -0.2
        ns     = N_STEPS_VALUES[:-1]

        print(f"T = {T:.4f}y")
        for n in ns:
            r = resids.get(n, 0.0)
            print(f"  max |log_price(n={n:4d}) - log_price(n=3200)| on k<-0.2:  {r:.2e}")

        # Convergence rate between first two entries
        r200 = resids.get(200, 0.0)
        r400 = resids.get(400, 0.0)
        if r200 > 0 and r400 > 0:
            rate = np.log2(r200 / r400)
        else:
            rate = float("nan")
        print(f"  convergence rate (n=200 vs n=400):  {rate:.2f}"
              "  (expect ~1 first-order, ~2 second-order ODE)")
        print(f"  wall time (n=200):  {wall_times[T][200]*1000:.0f} ms"
              f"  |  wall time (n=3200): {wall_times[T][3200]*1000:.0f} ms")

        # Update verdict flags
        if r200 < VERDICT_LARGE_RESID:
            all_under_resolved = False
        vals = [resids.get(n, 0.0) for n in ns]
        if max(vals) >= VERDICT_FLAT_RESID:
            all_flat = False
        # Check monotonicity
        for i in range(len(vals) - 1):
            if vals[i] < vals[i + 1] - 1e-15:
                any_non_monotone = True

        s["rate"] = rate

    # Shared convergence-rate check across all T
    rates = [s["rate"] for s in summary if np.isfinite(s.get("rate", float("nan")))]
    rate_ok = any(0.5 <= r <= 3.0 for r in rates) if rates else False

    print()
    r200_vals = [s["resids"].get(200, 0.0) for s in summary]
    if all(r > VERDICT_LARGE_RESID for r in r200_vals) and not any_non_monotone and rate_ok:
        ns_sufficient = None
        for n in N_STEPS_VALUES[:-1]:
            if all(s["resids"].get(n, 0.0) < 1e-4 for s in summary):
                ns_sufficient = n
                break
        if ns_sufficient is None:
            ns_sufficient = N_STEPS_VALUES[-1]
        print(
            "VERDICT: Riccati ODE is under-resolved at n_steps=200. "
            "Convergence rate consistent with finite-order ODE truncation error. "
            f"Recommend bumping default n_steps to the smallest value where the "
            f"residual drops below 1e-4 (n_steps = {ns_sufficient})."
        )
    elif all_flat:
        print(
            "VERDICT: Riccati ODE is NOT the cause. The deep-OTM 'wobble' is real "
            "model output, not a numerical artifact. Investigate the lifted-factor "
            "discretization (parameter n, the number of factors) or accept it as a "
            "property of the approximation."
        )
    else:
        print(
            "VERDICT: borderline / non-monotone convergence. Inspect the figure manually."
        )


if __name__ == "__main__":
    main()
