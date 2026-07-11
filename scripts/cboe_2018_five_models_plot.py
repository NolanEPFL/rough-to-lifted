"""scripts/cboe_2018_five_models_plot.py - aggregate + plot the lambda=inf calibrations.

Reads the five per-model JSONs written by cboe_2018_five_models_laminf.py and
produces:
  - a combined results JSON
  - a per-maturity CSV (T, empirical, lh-geo, lh-l2, ab-geo, ab-l2, tf-lh, paper)
  - a LINEAR-axes plot overlaying all five calibrated-model skew curves against
    the empirical points and the paper's reference power law.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
FIG_DIR = Path(_ROOT) / "figures" / "validation"
PLOT_PNG = FIG_DIR / "atm_skew_5models_laminf_2018-06-20.png"
PLOT_PNG_MIRROR = RES_DIR / "atm_skew_5models_laminf_2018-06-20.png"
COMBINED_JSON = RES_DIR / "five_models_laminf_combined.json"
COMBINED_CSV = RES_DIR / "five_models_laminf_per_maturity.csv"

MODELS = [
    ("lh-geo", "LH-geo",  "#1565c0", "-"),
    ("lh-l2",  "LH-L²", "#0277bd", "--"),
    ("ab-geo", "aB-geo",  "#c62828", "-"),
    ("ab-l2",  "aB-L²", "#ad1457", "--"),
    ("tf-lh",  "TF-LH",   "#2e7d32", "-"),
]
PAPER_C, PAPER_ALPHA = 0.35, -0.41


def _params_label(model_key, p):
    """Short legend label for the calibrated parameters."""
    if model_key in ("lh-geo", "lh-l2"):
        return f"H={p['H']:.3f}, $\\nu$={p['nu']:.3f}, $\\rho$={p['rho']:.3f}"
    if model_key in ("ab-geo", "ab-l2"):
        return f"H={p['H']:.3f}, $\\eta$={p['eta']:.3f}, $\\rho$={p['rho']:.3f}"
    if model_key == "tf-lh":
        return (f"H$_1$={p['H1']:.3f}, $\\nu_1$={p['nu1']:.3f}, $\\rho_1$={p['rho1']:.3f},  "
                f"$\\lambda_2$={p['lam2']:.2f}, $\\theta_2$={p['theta2']:.4f}, "
                f"$\\nu_2$={p['nu2']:.3f}, $\\rho_2$={p['rho2']:.3f}")
    return ""


def main() -> None:
    summaries = {}
    missing = []
    for key, _, _, _ in MODELS:
        p = RES_DIR / f"five_models_laminf_{key}.json"
        if not p.exists():
            missing.append(key)
            continue
        summaries[key] = json.loads(p.read_text())

    if missing:
        raise FileNotFoundError(f"Per-model JSONs missing: {missing}. "
                                 f"Run cboe_2018_five_models_laminf.py --model <name> first.")

    # Empirical column should match across all five (they share the same surface);
    # take it from lh-geo as canonical.
    canon = summaries["lh-geo"]
    df = pd.DataFrame({"T": [r["T"] for r in canon["skew_per_maturity"]],
                       "empirical_skew_spot": [r["empirical_skew_spot"]
                                                for r in canon["skew_per_maturity"]]})
    for key, _, _, _ in MODELS:
        col = f"model_skew_spot_{key}"
        df[col] = [r["model_skew_spot"] for r in summaries[key]["skew_per_maturity"]]

    df["paper_pred_0.35_T^-0.41"] = PAPER_C * df["T"].values ** PAPER_ALPHA

    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMBINED_CSV, index=False)
    print(f"[plot] wrote {COMBINED_CSV} ({len(df)} maturities)")

    combined = {
        "date": "2018-06-20",
        "objective": "lambda=inf (pure ATM-skew RMSE)",
        "S0_spot": 2769.59,
        "per_model": {
            key: {
                "calibrated_params": summaries[key]["calibrated_params"],
                "loss_at_optimum_skew_RMSE": summaries[key]["loss_at_optimum_skew_RMSE"],
                "elapsed_calibration_seconds": summaries[key]["elapsed_calibration_seconds"],
                "elapsed_total_seconds": summaries[key]["elapsed_total_seconds"],
                "n_evals": summaries[key]["n_evals"],
                "convergence_note": summaries[key]["convergence_note"],
            }
            for key, _, _, _ in MODELS
        },
        "paper_reference": {"C": PAPER_C, "alpha": PAPER_ALPHA},
    }
    COMBINED_JSON.write_text(json.dumps(combined, indent=2,
                                          default=lambda x: None if x != x else float(x)))
    print(f"[plot] wrote {COMBINED_JSON}")

    # --- Plot: linear axes ---
    T = df["T"].values
    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    # Empirical points
    ax.scatter(T, df["empirical_skew_spot"], s=46, color="black",
               marker="o", label="empirical (spot-moneyness)", zorder=5)

    # Five model lines
    for key, label, color, style in MODELS:
        col = f"model_skew_spot_{key}"
        s = df[col].values.astype(float)
        ok = np.isfinite(s) & (s > 0)
        plabel = _params_label(key, summaries[key]["calibrated_params"])
        loss = summaries[key]["loss_at_optimum_skew_RMSE"]
        ax.plot(T[ok], s[ok], style, color=color, lw=1.7,
                label=f"{label}  ({plabel}; skew-RMSE={loss:.3f})", zorder=4)

    # Paper reference
    T_dense = np.linspace(max(T.min(), 0.005), T.max(), 600)
    ax.plot(T_dense, PAPER_C * T_dense ** PAPER_ALPHA, ":", color="#666666", lw=1.2,
            label=r"Abi Jaber (2019) Fig. 1:  $0.35\,T^{-0.41}$", zorder=2)

    ax.set_xlim(0.0, T.max() * 1.02)
    y_max = float(np.nanmax([
        np.nanmax(df["empirical_skew_spot"].values),
        np.nanmax(df["paper_pred_0.35_T^-0.41"].values),
        *[np.nanmax(df[f"model_skew_spot_{key}"].values) for key, _, _, _ in MODELS
          if df[f"model_skew_spot_{key}"].notna().any()]
    ]))
    ax.set_ylim(0.0, y_max * 1.07)
    ax.set_xlabel("maturity $T$ (years)")
    ax.set_ylabel("ATM skew 98/102 (spot-moneyness)")
    ax.set_title(r"2018-06-20 ATM skew: five models calibrated at $\lambda=\infty$ "
                  "(skew objective)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.95)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=140, bbox_inches="tight")
    fig.savefig(PLOT_PNG_MIRROR, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {PLOT_PNG}")
    print(f"[plot] wrote {PLOT_PNG_MIRROR} (mirror)")

    print()
    print("=== PER-MODEL SUMMARY ===")
    for key, label, _, _ in MODELS:
        s = summaries[key]
        print(f"  {label:10s}  loss={s['loss_at_optimum_skew_RMSE']:.4f}  "
              f"t={s['elapsed_calibration_seconds']:.0f}s  evals={s['n_evals']}  "
              f"flag={s['convergence_note']}")
        print(f"    params: {json.dumps(s['calibrated_params'])}")


if __name__ == "__main__":
    main()
