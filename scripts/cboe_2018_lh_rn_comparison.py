"""scripts/cboe_2018_lh_rn_comparison.py - side-by-side r_n=2.5 vs r_n=1.674.

Loads the two lambda=0 LH-geo calibration outputs on 2018-06-20:
  - lh_calibration_2018-06-20.json + lh_calibration_skew_per_maturity.csv
    (r_n = 2.5, my thesis default)
  - lh_calibration_paper_rn_2018-06-20.json + lh_calibration_paper_rn_skew_per_maturity.csv
    (r_n = 1 + 10 * 20^(-0.9) ≈ 1.6740, paper's geometric grid)
and produces:
  - a JSON side-by-side summary,
  - a CSV with per-maturity (empirical, model_r_n_2.5, model_r_n_paper, paper),
  - an overlay plot (4 series).
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


OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
J_25 = OUT_DIR / "lh_calibration_2018-06-20.json"
J_PR = OUT_DIR / "lh_calibration_paper_rn_2018-06-20.json"
S_25 = OUT_DIR / "lh_calibration_skew_per_maturity.csv"
S_PR = OUT_DIR / "lh_calibration_paper_rn_skew_per_maturity.csv"
OUT_JSON = OUT_DIR / "lh_rn_comparison.json"
OUT_CSV = OUT_DIR / "lh_rn_comparison_per_maturity.csv"
OUT_PLOT = OUT_DIR / "lh_rn_comparison_skew.png"


def main() -> None:
    for p in (J_25, J_PR, S_25, S_PR):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    j25 = json.loads(J_25.read_text())
    jpr = json.loads(J_PR.read_text())
    s25 = pd.read_csv(S_25)
    spr = pd.read_csv(S_PR)

    # Merge skew tables on T
    merged = s25.merge(spr, on="T", suffixes=("_r_n_2.5", "_r_n_paper"))
    # The empirical and paper columns are identical between the two files - keep
    # one copy each (using the r_n_2.5 versions as canonical).
    merged = merged.rename(columns={
        "empirical_skew_spot_r_n_2.5": "empirical_skew_spot",
        "paper_pred_0.35_T^-0.41_r_n_2.5": "paper_pred_0.35_T^-0.41",
    }).drop(columns=[
        "empirical_skew_spot_r_n_2.5".replace("_r_n_2.5", "_r_n_paper").replace("empirical", "empirical"),
    ], errors="ignore")
    if "empirical_skew_spot_r_n_paper" in merged.columns:
        merged = merged.drop(columns=["empirical_skew_spot_r_n_paper"])
    if "paper_pred_0.35_T^-0.41_r_n_paper" in merged.columns:
        merged = merged.drop(columns=["paper_pred_0.35_T^-0.41_r_n_paper"])
    # Rename model_skew columns for clarity
    merged = merged.rename(columns={
        "model_skew_spot_r_n_2.5": "model_skew_r_n_2.5",
        "model_skew_spot_r_n_paper": "model_skew_r_n_paper",
    })
    merged = merged[[
        "T", "empirical_skew_spot",
        "model_skew_r_n_2.5", "model_skew_r_n_paper",
        "paper_pred_0.35_T^-0.41",
    ]].sort_values("T").reset_index(drop=True)
    merged["delta_2.5_minus_empirical"] = merged["model_skew_r_n_2.5"] - merged["empirical_skew_spot"]
    merged["delta_paper_minus_empirical"] = merged["model_skew_r_n_paper"] - merged["empirical_skew_spot"]
    merged["delta_2.5_minus_paperFn"] = merged["model_skew_r_n_2.5"] - merged["paper_pred_0.35_T^-0.41"]
    merged["delta_paper_minus_paperFn"] = merged["model_skew_r_n_paper"] - merged["paper_pred_0.35_T^-0.41"]
    merged.to_csv(OUT_CSV, index=False)
    print(f"[compare] wrote {OUT_CSV}")

    # Side-by-side numerical summary
    def _g(d, *path):
        cur = d
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur

    summary = {
        "r_n_2.5": {
            "r_n": _g(j25, "config_used", "r_n"),
            "calibrated_params": j25["calibrated_params"],
            "H_at_lower_bound": abs(j25["calibrated_params"]["H"] - 0.02) < 1e-6,
            "iv_rmse_thin": _g(j25, "iv_rmse_thin_grid_calibration"),
            "iv_rmse_full": _g(j25, "iv_rmse_full_surface_diagnostic"),
            "elapsed_seconds": j25.get("elapsed_seconds"),
            "n_evals": j25.get("n_evals"),
            "loss_at_optimum": j25.get("loss_at_optimum_vega_RMSE"),
        },
        "r_n_paper": {
            "r_n": _g(jpr, "config_used", "r_n"),
            "r_n_formula": _g(jpr, "config_used", "r_n_formula"),
            "calibrated_params": jpr["calibrated_params"],
            "H_at_lower_bound": abs(jpr["calibrated_params"]["H"] - 0.02) < 1e-6,
            "iv_rmse_thin": _g(jpr, "iv_rmse_thin_grid_calibration"),
            "iv_rmse_full": _g(jpr, "iv_rmse_full_surface_diagnostic"),
            "elapsed_seconds": jpr.get("elapsed_seconds"),
            "n_evals": jpr.get("n_evals"),
            "loss_at_optimum": jpr.get("loss_at_optimum_vega_RMSE"),
        },
        "skew_short_end_T_under_3m": {
            "n_maturities": int((merged["T"] < 90 / 365).sum()),
            "mean_model_minus_empirical_r_n_2.5":
                float(merged.loc[merged["T"] < 90 / 365, "delta_2.5_minus_empirical"].mean()),
            "mean_model_minus_empirical_r_n_paper":
                float(merged.loc[merged["T"] < 90 / 365, "delta_paper_minus_empirical"].mean()),
            "max_model_minus_empirical_r_n_2.5":
                float(merged.loc[merged["T"] < 90 / 365, "delta_2.5_minus_empirical"].max()),
            "max_model_minus_empirical_r_n_paper":
                float(merged.loc[merged["T"] < 90 / 365, "delta_paper_minus_empirical"].max()),
        },
        "skew_long_end_T_over_1y": {
            "n_maturities": int((merged["T"] > 1.0).sum()),
            "mean_model_minus_empirical_r_n_2.5":
                float(merged.loc[merged["T"] > 1.0, "delta_2.5_minus_empirical"].mean()),
            "mean_model_minus_empirical_r_n_paper":
                float(merged.loc[merged["T"] > 1.0, "delta_paper_minus_empirical"].mean()),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[compare] wrote {OUT_JSON}")

    # Plot: empirical, model r_n=2.5, model r_n=paper, paper power law
    T = merged["T"].values
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.scatter(T, merged["empirical_skew_spot"], s=44, color="#c62828", marker="o",
               label="empirical (spot-moneyness)", zorder=5)
    m25 = merged["model_skew_r_n_2.5"].values
    mpr = merged["model_skew_r_n_paper"].values
    ok25 = np.isfinite(m25) & (m25 > 0)
    okpr = np.isfinite(mpr) & (mpr > 0)
    p25 = j25["calibrated_params"]; ppr = jpr["calibrated_params"]
    ax.plot(T[ok25], m25[ok25], "-", color="#1565c0", lw=1.7,
            label=(f"LH-geo  r_n=2.500  (H={p25['H']:.3f}, "
                   f"nu={p25['nu']:.3f}, rho={p25['rho']:.3f})"), zorder=4)
    ax.plot(T[okpr], mpr[okpr], "-", color="#2e7d32", lw=1.7,
            label=(f"LH-geo  r_n={jpr['config_used']['r_n']:.4f} (paper grid)  "
                   f"(H={ppr['H']:.3f}, nu={ppr['nu']:.3f}, rho={ppr['rho']:.3f})"), zorder=4)
    T_dense = np.linspace(T.min(), T.max(), 400)
    ax.plot(T_dense, 0.35 * T_dense ** -0.41, "--", color="black", lw=1.3,
            label="Abi Jaber (2019) Fig.1:  0.35 * T^(-0.41)", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("ATM skew 98/102")
    ax.set_title("2018-06-20: r_n=2.5 vs paper grid r_n=1.6740 (everything else fixed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {OUT_PLOT}")

    # Console summary
    print()
    print("=== SIDE-BY-SIDE SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
