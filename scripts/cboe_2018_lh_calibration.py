"""scripts/cboe_2018_lh_calibration.py - Step 3 of the 2018 validation.

Calibrate LH-geo at lambda=0 (pure vega-weighted IV-RMSE) on the 2018-06-20 SPX
surface using my existing calibration pipeline + the SAME thinning convention
the three-pairs work uses (scripts/07_temporal_oos.py defaults:
max_maturities=8 log-spaced, max_strikes=15 per maturity). Then compare the
calibrated model's 98/102 ATM skew (spot-moneyness, same estimator as Step 2) to:
  (i)  the empirical skew from Step 2 (the red dots), and
  (ii) Abi Jaber (2019) Figure 1's reference power law 0.35 * T^(-0.41).

Calibration is on the THIN grid (~120 points) to match the three-pairs convention;
the diagnostic IV-RMSE and the model-vs-empirical skew comparison are evaluated on
the FULL cleaned surface. Single-date: no train/test split.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.calibration.loss import IVSurfaceLoss, _atm_skew_98_102
from src.calibration.optimizer import calibrate_lifted_heston
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.common.black_scholes import bs_vega
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface


SURFACE_CSV = Path(_ROOT) / "data" / "spx_2018-06-20.csv"
OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
RESULT_JSON = OUT_DIR / "lh_calibration_2018-06-20.json"
SKEW_CSV = OUT_DIR / "lh_calibration_skew_per_maturity.csv"
PLOT_PNG = OUT_DIR / "lh_calib_vs_empirical_vs_paper.png"

S0 = 2769.59
PAPER_C = 0.35
PAPER_ALPHA = -0.41

# Match scripts/07_temporal_oos.py defaults exactly
MAX_MATURITIES = 8
MAX_STRIKES = 15


def _thin(market_ivs, strikes_per, forwards_per, max_mats, max_strikes):
    """Same thinning logic as scripts/07_temporal_oos.py._thin."""
    all_T = np.array(sorted(market_ivs.keys()))
    if max_mats and len(all_T) > max_mats:
        targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), max_mats))
        idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
        all_T = all_T[sorted(set(idx))]
    new_iv, new_K, new_F = {}, {}, {}
    for T in all_T:
        ivs = market_ivs[T]; Ks = strikes_per[T]
        if max_strikes and len(Ks) > max_strikes:
            idx = np.round(np.linspace(0, len(Ks) - 1, max_strikes)).astype(int)
            Ks = Ks[idx]; ivs = ivs[idx]
        new_iv[float(T)] = ivs
        new_K[float(T)]  = Ks
        new_F[float(T)]  = forwards_per.get(float(T), 0.0)
    return new_iv, new_K, new_F


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[step3] loading SPXSurface from {SURFACE_CSV}")
    surf = load_spx_csv(SURFACE_CSV, weight_scheme="vega")

    # FULL cleaned surface (for diagnostic IV-RMSE + model skew evaluation)
    K_full = surf.strikes_per_T()
    mkt_full = surf.ivs_per_T()
    F_full = surf.forward_per_T()
    n_full = int(sum(len(v) for v in mkt_full.values()))
    print(f"[step3] FULL surface: {len(K_full)} maturities, {n_full} (T, K) points")

    # THIN grid for calibration (matches scripts/07_temporal_oos.py defaults)
    mkt, K, F = _thin(mkt_full, K_full, F_full, MAX_MATURITIES, MAX_STRIKES)
    n_thin = int(sum(len(v) for v in mkt.values()))
    print(f"[step3] THIN  surface (calibration): {len(K)} maturities, {n_thin} (T, K) points "
          f"(max_maturities={MAX_MATURITIES}, max_strikes={MAX_STRIKES})")

    # Forward variance curve from ATM total variance (uses FULL surface)
    xi0_mats, xi0_vals = fit_xi0_from_surface(surf)
    fv = PiecewiseConstantForwardVariance(xi0_mats, xi0_vals)
    print(f"[step3] xi0 grid: {len(xi0_mats)} knots, "
          f"xi0(0)={float(xi0_vals[0]):.5f}, xi0_max={float(np.max(xi0_vals)):.5f}")

    # Loss at lambda = 0 on the THIN grid
    loss = IVSurfaceLoss(
        market_ivs=mkt, strikes=K, S0=S0, forwards=F,
        weights="vega", skew_lambda=0.0, in_sample_T_max=None,
    )

    print("[step3] calibrating LH-geo (n=20, r_n=2.5, kernel=geometric, lambda=0, seed=42) ...")
    t0 = time.time()
    res = calibrate_lifted_heston(
        loss, fv, n=20, r_n=2.5, kernel="geometric", seed=42,
    )
    elapsed = time.time() - t0
    print(f"[step3] done in {elapsed:.1f}s, n_evals={res.n_evals}, "
          f"loss_at_optimum={res.loss_in_sample:.6f}")
    print(f"[step3] calibrated params: {res.params}")

    # Build calibrated model + evaluate on the FULL cleaned-surface strikes
    params = LiftedHestonParams(
        H=res.params["H"], n=20, r_n=2.5, kernel="geometric",
        nu=res.params["nu"], rho=res.params["rho"],
    )
    iv_model_full = lifted_heston_iv_surface(params, fv, S0, K_full)

    # Diagnostic IV-RMSE on the FULL surface (NaN-filtered: the Riccati overflows
    # at a handful of extreme deep-OTM strikes => model IV is NaN there; we skip
    # those rows, count them, and report the fraction).
    sse_vw = 0.0; weight_sum = 0.0
    sse_plain = 0.0; count_used = 0; count_total = 0; n_nan_full = 0
    # Thin IV-RMSE separately (the actual training set)
    sse_vw_thin = 0.0; weight_sum_thin = 0.0
    sse_plain_thin = 0.0; count_thin = 0

    for T_val in sorted(K_full.keys()):
        K_T = K_full[T_val]
        ivs_mkt = np.asarray(mkt_full[T_val], dtype=float)
        ivs_mdl = np.asarray(iv_model_full[T_val], dtype=float)
        F_T = float(F_full[T_val])
        count_total += len(ivs_mkt)
        mask = np.isfinite(ivs_mkt) & np.isfinite(ivs_mdl)
        n_nan_full += int((~mask).sum())
        if not mask.any():
            continue
        K_T_f = K_T[mask]; ivs_mkt_f = ivs_mkt[mask]; ivs_mdl_f = ivs_mdl[mask]
        vegas = np.array([bs_vega(F_T, k, T_val, iv) for k, iv in zip(K_T_f, ivs_mkt_f)])
        diff = ivs_mkt_f - ivs_mdl_f
        sse_vw += float(np.sum(vegas * diff ** 2)); weight_sum += float(np.sum(vegas))
        sse_plain += float(np.sum(diff ** 2)); count_used += int(len(diff))
    rmse_vw_full = float(np.sqrt(sse_vw / weight_sum)) if weight_sum > 0 else float("nan")
    rmse_plain_full = float(np.sqrt(sse_plain / count_used)) if count_used > 0 else float("nan")

    # IV-RMSE on the THIN grid (the actual calibration objective)
    iv_model_thin = lifted_heston_iv_surface(params, fv, S0, K)
    for T_val in sorted(K.keys()):
        K_T = K[T_val]
        ivs_mkt = np.asarray(mkt[T_val], dtype=float)
        ivs_mdl = np.asarray(iv_model_thin[T_val], dtype=float)
        F_T = float(F[T_val])
        vegas = np.array([bs_vega(F_T, k, T_val, iv) for k, iv in zip(K_T, ivs_mkt)])
        diff = ivs_mkt - ivs_mdl
        sse_vw_thin += float(np.sum(vegas * diff ** 2)); weight_sum_thin += float(np.sum(vegas))
        sse_plain_thin += float(np.sum(diff ** 2)); count_thin += int(len(diff))
    rmse_vw_thin = float(np.sqrt(sse_vw_thin / weight_sum_thin)) if weight_sum_thin > 0 else float("nan")
    rmse_plain_thin = float(np.sqrt(sse_plain_thin / count_thin)) if count_thin > 0 else float("nan")

    print(f"[step3] IV-RMSE on THIN grid (calibration): vega-w={rmse_vw_thin:.5f}, "
          f"plain={rmse_plain_thin:.5f}, n={count_thin}")
    print(f"[step3] IV-RMSE on FULL surface (diagnostic): vega-w={rmse_vw_full:.5f}, "
          f"plain={rmse_plain_full:.5f}, n_used={count_used}/{count_total} "
          f"(model NaN on {n_nan_full} deep-OTM rows)")

    # ATM skew per maturity (spot-moneyness, same estimator as Step 2) on FULL surface
    skew_rows = []
    for T_val in sorted(K_full.keys()):
        s_emp = _atm_skew_98_102(S0, K_full[T_val], np.asarray(mkt_full[T_val], dtype=float))
        s_mdl = _atm_skew_98_102(S0, K_full[T_val], np.asarray(iv_model_full[T_val], dtype=float))
        skew_rows.append({
            "T": float(T_val),
            "empirical_skew_spot": (None if s_emp is None else float(s_emp)),
            "model_skew_spot": (None if s_mdl is None else float(s_mdl)),
            "paper_pred_0.35_T^-0.41": float(PAPER_C * T_val ** PAPER_ALPHA),
        })
    pd.DataFrame(skew_rows).to_csv(SKEW_CSV, index=False)
    print(f"[step3] wrote per-maturity skew CSV: {SKEW_CSV}")

    summary = {
        "config_used": {
            "n_factors": 20, "r_n": 2.5, "kernel": "geometric",
            "skew_lambda": 0.0, "weights": "vega",
            "free_params": ["nu", "rho", "H"],
            "lambda_mr": 0.0, "xi0_source": "ATM total variance (PCHIP)",
            "thinning": {"max_maturities": MAX_MATURITIES, "max_strikes": MAX_STRIKES,
                          "source": "scripts/07_temporal_oos.py defaults"},
            "seed": 42,
        },
        "calibrated_params": {
            "H": float(res.params["H"]),
            "nu": float(res.params["nu"]),
            "rho": float(res.params["rho"]),
        },
        "bounds_LH": {"nu": [0.05, 3.0], "rho": [-0.99, 0.0], "H": [0.02, 0.49]},
        "loss_at_optimum_vega_RMSE": float(res.loss_in_sample),
        "iv_rmse_thin_grid_calibration": {
            "vega_weighted": rmse_vw_thin, "plain": rmse_plain_thin,
            "n_points": count_thin, "n_maturities": len(K),
        },
        "iv_rmse_full_surface_diagnostic": {
            "vega_weighted": rmse_vw_full, "plain": rmse_plain_full,
            "n_points_used": count_used, "n_points_total": count_total,
            "n_model_iv_nan": n_nan_full, "n_maturities": len(K_full),
            "note": "NaN-filtered: Riccati overflows at extreme deep-OTM strikes",
        },
        "elapsed_seconds": float(elapsed),
        "n_evals": int(res.n_evals),
        "xi0_anchored_at_0": float(xi0_vals[0]),
        "paper_reference": {"C": PAPER_C, "alpha": PAPER_ALPHA},
        "skew_per_maturity_path": str(SKEW_CSV.relative_to(_ROOT)),
    }
    RESULT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[step3] wrote summary JSON: {RESULT_JSON}")

    # Plot: empirical (red dots), calibrated model (blue line), paper (black dashed)
    Ts = np.array([r["T"] for r in skew_rows])
    s_emp = np.array([r["empirical_skew_spot"] if r["empirical_skew_spot"] is not None else np.nan
                      for r in skew_rows])
    s_mdl = np.array([r["model_skew_spot"] if r["model_skew_spot"] is not None else np.nan
                      for r in skew_rows])

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(Ts, s_emp, s=44, color="#c62828", marker="o",
               label="empirical (spot-moneyness, from Step 2)", zorder=4)
    ok_mdl = np.isfinite(s_mdl) & (s_mdl > 0)
    ax.plot(Ts[ok_mdl], s_mdl[ok_mdl], "-", color="#1565c0", lw=1.6,
            label=(f"calibrated LH-geo  (H={params.H:.3f}, "
                   f"nu={params.nu:.3f}, rho={params.rho:.3f})"), zorder=3)
    ax.scatter(Ts[ok_mdl], s_mdl[ok_mdl], s=22, color="#1565c0", marker="x", zorder=3)

    T_dense = np.linspace(Ts.min(), Ts.max(), 400)
    paper = PAPER_C * T_dense ** PAPER_ALPHA
    ax.plot(T_dense, paper, "--", color="black", lw=1.3,
            label="Abi Jaber (2019) Fig.1:  0.35 * T^(-0.41)", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("ATM skew 98/102")
    ax.set_title("2018-06-20: calibrated LH-geo  vs  empirical  vs  Abi Jaber Fig.1\n"
                 "(lambda=0 calibration, ATM-IV ξ₀ fit, spot-moneyness 98/102 skew)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=140)
    plt.close(fig)
    print(f"[step3] wrote plot: {PLOT_PNG}")

    print()
    print("=== STEP 3 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
