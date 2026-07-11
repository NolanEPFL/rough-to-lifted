"""Diagnostic: is the TF-LH deep-OTM wobble in the prices or in the BS inversion?

Previous diagnostic (_diag_ncos.py) ruled out COS resolution. This script
isolates whether the wobble originates in the raw call prices or in the
Black-Scholes IV inversion of near-zero prices.

Usage
-----
    python scripts/_diag_price_vs_iv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.common.black_scholes import bs_call_price, bs_implied_vol
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import two_factor_lh_call_prices

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CSV  = _ROOT / "data" / "spx_2024-12-04.csv"
JSON_PATH = (_ROOT / "results/07b_temporal_oos_two_factor"
             / "2024-12-04_2024-12-05_42_lam0.0/temporal_oos_results.json")
OUT_DIR   = _ROOT / "results/_diag_price_vs_iv"

T_DIAG_TARGETS = [0.293, 0.619, 1.537]
N_COS, L0, N_STEPS = 256, 12.0, 200
WING_K_MAX = -0.2    # deep OTM region: k < this
POLY_DEG   = 4


def poly_smooth(k: np.ndarray, y: np.ndarray, k_eval: np.ndarray) -> np.ndarray:
    """Fit degree-4 polynomial to (k, y) and evaluate at k_eval."""
    if len(k) < POLY_DEG + 1:
        return np.full_like(k_eval, np.nan)
    coeffs = np.polyfit(k, y, POLY_DEG)
    return np.polyval(coeffs, k_eval)


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

    # ── Closest T for each target ─────────────────────────────────────────────
    T_diag = [float(all_T[np.argmin(np.abs(all_T - t))]) for t in T_DIAG_TARGETS]
    print(f"Diagnostic maturities: {[f'{t:.4f}' for t in T_diag]}")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    fig.suptitle("TF-LH price vs IV diagnostic — deep OTM put wing (k < −0.2)",
                 fontsize=13)

    summary: list[dict] = []

    for row, T in enumerate(T_diag):
        K_arr = K_per_T[T]
        F     = F_per_T[T]
        iv_m  = iv_mkt[T]

        k_arr = np.log(K_arr / F)
        order = np.argsort(k_arr)
        k_s   = k_arr[order]
        K_s   = K_arr[order]
        iv_m_s = iv_m[order]

        # ── Model call prices ─────────────────────────────────────────────────
        mdl_prices_raw = two_factor_lh_call_prices(
            params, fv, S0, K_s, T, N_cos=N_COS, L0=L0, n_steps=N_STEPS
        )
        mdl_prices = np.maximum(mdl_prices_raw, 0.0)

        # ── Market call prices from market IVs (clean reference) ──────────────
        mkt_prices = np.array([
            bs_call_price(F, float(K), T, float(iv))
            if np.isfinite(iv) and iv > 0 else np.nan
            for K, iv in zip(K_s, iv_m_s)
        ])

        # ── Model IV from model prices ────────────────────────────────────────
        mdl_iv = np.array([
            bs_implied_vol(float(p_), F, float(K), T, option_type="C")
            for p_, K in zip(mdl_prices, K_s)
        ])

        # ── Deep OTM wing mask ────────────────────────────────────────────────
        wing = k_s < WING_K_MAX
        k_w  = k_s[wing]
        mdl_p_w  = mdl_prices[wing]
        mdl_iv_w = mdl_iv[wing]
        iv_m_w   = iv_m_s[wing]

        # ── Log-price residual ────────────────────────────────────────────────
        # Only on points where model price is positive
        pos = wing & (mdl_prices > 0)
        k_pos   = k_s[pos]
        log_p   = np.log(mdl_prices[pos] + 1e-300)
        if len(k_pos) >= POLY_DEG + 1:
            log_p_smooth = poly_smooth(k_pos, log_p, k_pos)
            resid_log_p  = log_p - log_p_smooth
        else:
            log_p_smooth = np.full_like(k_pos, np.nan)
            resid_log_p  = np.full_like(k_pos, np.nan)

        # ── IV residual ───────────────────────────────────────────────────────
        fin_iv = wing & np.isfinite(mdl_iv) & (mdl_iv > 0.01) & (mdl_iv < 1.5)
        k_iv   = k_s[fin_iv]
        iv_w_f = mdl_iv[fin_iv]
        if len(k_iv) >= POLY_DEG + 1:
            iv_smooth  = poly_smooth(k_iv, iv_w_f, k_iv)
            resid_iv   = (iv_w_f - iv_smooth) * 10000   # bps
        else:
            iv_smooth = np.full_like(k_iv, np.nan)
            resid_iv  = np.full_like(k_iv, np.nan)

        # ── Numerical summary values ──────────────────────────────────────────
        max_resid_log_p = float(np.max(np.abs(resid_log_p))) if len(resid_log_p) > 0 else 0.0
        max_resid_iv    = float(np.max(np.abs(resid_iv)))    if len(resid_iv) > 0    else 0.0
        amp = max_resid_iv / max(max_resid_log_p, 1e-20) if max_resid_log_p > 0 else float("inf")
        summary.append(dict(T=T, max_log_p=max_resid_log_p, max_iv=max_resid_iv, amp=amp))

        # ── Col 0: model price vs k, log y-axis, with poly smoother ──────────
        ax = axes[row, 0]
        if len(k_pos) > 0:
            ax.semilogy(k_pos, np.exp(log_p),      "b-", lw=1.5, label="Model price")
            ax.semilogy(k_pos, np.exp(log_p_smooth), "r--", lw=1.2, label="Poly fit")
        fin_mkt_w = wing & np.isfinite(mkt_prices) & (mkt_prices > 0)
        if fin_mkt_w.any():
            ax.semilogy(k_s[fin_mkt_w], mkt_prices[fin_mkt_w],
                        "k.", ms=4, label="Mkt price")
        ax.set_title(f"T={T:.4f}y  prices (log)", fontsize=9)
        ax.set_xlabel("log(K/F)"); ax.set_ylabel("Call price")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # ── Col 1: residual log-price ─────────────────────────────────────────
        ax = axes[row, 1]
        if len(k_pos) > 0:
            ax.plot(k_pos, resid_log_p, "b-", lw=1.2)
            ax.axhline(0, color="black", lw=0.5, ls="--")
            ax.axhline(1e-4,  color="red", lw=0.5, ls=":", label="±1e-4")
            ax.axhline(-1e-4, color="red", lw=0.5, ls=":")
        ax.set_title(f"T={T:.4f}y  log-price residual", fontsize=9)
        ax.set_xlabel("log(K/F)"); ax.set_ylabel("log(price) – poly")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # ── Col 2: model IV vs k with market dots ─────────────────────────────
        ax = axes[row, 2]
        if fin_iv.any():
            ax.plot(k_iv, iv_w_f, "b-", lw=1.5, label="Model IV")
        fin_mkt_iv = wing & np.isfinite(iv_m_s) & (iv_m_s > 0.01) & (iv_m_s < 1.5)
        if fin_mkt_iv.any():
            ax.scatter(k_s[fin_mkt_iv], iv_m_s[fin_mkt_iv],
                       s=18, color="black", zorder=5, label="Market IV")
        ax.set_title(f"T={T:.4f}y  model IV", fontsize=9)
        ax.set_xlabel("log(K/F)"); ax.set_ylabel("IV")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # ── Col 3: IV residual (wobble in IV space) ───────────────────────────
        ax = axes[row, 3]
        if len(k_iv) > 0:
            ax.plot(k_iv, resid_iv, "b-", lw=1.2)
            ax.axhline(0,   color="black", lw=0.5, ls="--")
            ax.axhline(10,  color="orange", lw=0.5, ls=":", label="±10 bps")
            ax.axhline(-10, color="orange", lw=0.5, ls=":")
        ax.set_title(f"T={T:.4f}y  IV residual (bps)", fontsize=9)
        ax.set_xlabel("log(K/F)"); ax.set_ylabel("IV – poly (bps)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    out_path = OUT_DIR / "price_vs_iv.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")

    # ── Numerical summary ─────────────────────────────────────────────────────
    print()
    all_log_p_smooth = True
    all_iv_large     = True
    any_log_p_large  = False

    for s in summary:
        print(f"T = {s['T']:.4f}y")
        print(f"  max |residual_log_price| on k<-0.2:   {s['max_log_p']:.2e}")
        print(f"  max |residual_IV        | on k<-0.2:  {s['max_iv']:.1f} bps")
        print(f"  amplification factor (IV/price):      {s['amp']:.1f}x")
        if s["max_log_p"] >= 1e-5:
            all_log_p_smooth = False
        if s["max_iv"] <= 10:
            all_iv_large = False
        if s["max_log_p"] > 1e-4:
            any_log_p_large = True

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    if all_log_p_smooth and all_iv_large:
        print("VERDICT: prices are smooth; the wobble enters via BS IV inversion of "
              "near-zero prices. The fix is to mask strikes with model price below a "
              "numerical floor (e.g. 1e-10 * S0) before inversion.")
    elif any_log_p_large:
        print("VERDICT: the wobble is upstream of inversion. Investigate the "
              "characteristic function and/or Riccati ODE next.")
    else:
        print("VERDICT: borderline. Both prices and IVs show some non-smoothness. "
              "Investigate both.")


if __name__ == "__main__":
    main()
