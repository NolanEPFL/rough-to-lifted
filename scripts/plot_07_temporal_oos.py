"""scripts/plot_07_temporal_oos.py — Visualise temporal OOS results.

Loads the JSON produced by 07_temporal_oos.py, re-prices all models on
both surfaces, and generates four figures:

  Figure 1 — IV slices on TRAIN date (IS fit quality)
  Figure 2 — IV slices on TEST  date (temporal OOS quality)
  Figure 3 — ATM skew (98/102) on both dates
  Figure 4 — IS vs temporal-OOS bar chart

Usage
-----
    python scripts/plot_07_temporal_oos.py
    python scripts/plot_07_temporal_oos.py --dir results/07_temporal_oos/2024-08-05_2024-08-06_42_lam0.0
    python scripts/plot_07_temporal_oos.py --no-ab   # skip slow aB repricing
"""
from __future__ import annotations

import argparse
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
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.pricing import abergomi_iv_surface


# ── Style — identical to notebooks/spx_analysis.ipynb ────────────────────────
STYLES = {
    "LH-geo": dict(color="blue",  ls="-",  lw=1.5, label="LH-geo"),
    "LH-L2":  dict(color="blue",  ls="--", lw=1.2, label="LH-L²"),
    "aB-L2":  dict(color="red",   ls="--", lw=1.5, label="aB-L²"),
    "aB-geo": dict(color="green", ls=":",  lw=1.5, label="aB-geo"),
}
MKT_STYLE = dict(color="black", marker="o", ms=3.5, lw=0, zorder=5, label="Market")

T_MIN_PLOT = 1.0 / 52   # T < 1 week excluded (paper convention, ≈ 0.0192y)


def _load_surface(date: str, data_dir: Path):
    csv = data_dir / f"spx_{date}.csv"
    surf = load_spx_csv(csv)
    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surf.forwards.mean())
    return surf, fv, S0


def _reprice_lh(p, kernel, n, r_n, fv, S0, strikes_per_T):
    params = LiftedHestonParams(H=p["H"], n=n, r_n=r_n,
                                 nu=p["nu"], rho=p["rho"], kernel=kernel)
    return lifted_heston_iv_surface(params, fv, S0, strikes_per_T)


def _reprice_ab(p, kernel, n, r_n, fv, S0, strikes_per_T, M_paths, seed):
    from src.calibration.optimizer import _build_ab_params
    params = _build_ab_params(p["H"], n, p["eta"], p["rho"], kernel, r_n)
    ivs, _ = abergomi_iv_surface(params, fv, S0, strikes_per_T,
                                  M_paths=M_paths, qmc=False, seed=seed)
    return ivs


def _atm_skew(ivs_dict, strikes_dict, forwards_dict, rmse_iv_max: float = 0.03):
    """ATM skew term structure |d sigma_impl/dk|_{k=0}| via SVI fit per slice.

    Slices where the SVI unweighted rmse_iv > rmse_iv_max are excluded (NaN):
    those fits failed to converge to a sensible smile and the ATM derivative
    would be unreliable. The same threshold applies to market and all models.
    """
    from src.data.svi import fit_svi_from_iv_dict
    fits = fit_svi_from_iv_dict(ivs_dict, strikes_dict, forwards_dict)
    Ts, Sk = [], []
    for t in sorted(fits):
        if t < T_MIN_PLOT:
            continue
        fit = fits[t]
        if fit.rmse_iv > rmse_iv_max:
            continue
        Ts.append(t)
        Sk.append(abs(fit.atm_skew_derivative()))
    return np.array(Ts), np.array(Sk)


def _pick_maturities(all_T, n=6):
    """Log-spaced selection across the full maturity range."""
    stable = np.array(sorted(t for t in all_T if t >= T_MIN_PLOT))
    if len(stable) == 0:
        return stable
    targets = np.exp(np.linspace(np.log(stable[0]), np.log(stable[-1]), n))
    idx = [int(np.argmin(np.abs(stable - t))) for t in targets]
    return stable[sorted(set(idx))]


def _iv_slice_panel(title, mkt_ivs, mkt_K, mkt_F, model_ivs_dict, S0, T_plot):
    """IV slices: x-axis = log(K/F), one panel per maturity."""
    n = len(T_plot)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13)

    for ax, T in zip(axes[:n], T_plot):
        Ks = mkt_K.get(T)
        if Ks is None:
            ax.set_visible(False); continue
        F = mkt_F.get(T, S0)
        k = np.log(Ks / F)
        order = np.argsort(k)              # FIX
        k_s = k[order]
        assert np.all(np.diff(k_s) >= 0), "log-moneyness must be sorted before plotting"
        ax.plot(k_s, mkt_ivs[T][order], **MKT_STYLE)
        for name, ivs_all in model_ivs_dict.items():
            if T in ivs_all:
                iv_mdl = ivs_all[T][order] # apply same ordering
                fin = np.isfinite(iv_mdl) & (iv_mdl > 0.01) & (iv_mdl < 1.5)
                if fin.sum() > 1:
                    style = STYLES.get(name, {})
                    ax.plot(k_s[fin], iv_mdl[fin], **style)
        ax.set_title(f"T = {T:.3f}y", fontsize=9)
        ax.set_xlabel("log(K/F)")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(STYLES) + 1,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)

    for ax in axes[n:]:
        ax.set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     type=Path,
                        help="Path to results dir (default: most recent 07_temporal_oos)")
    parser.add_argument("--no-ab",   action="store_true",
                        help="Skip aB repricing (slow MC)")
    parser.add_argument("--M-paths", type=int, default=100_000)
    parser.add_argument("--data-dir",type=Path, default=_ROOT / "data")
    args = parser.parse_args()

    # ── Find results dir ──────────────────────────────────────────────────────
    if args.dir:
        result_dir = args.dir
    else:
        base = _ROOT / "results/07_temporal_oos"
        dirs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime)
        if not dirs:
            print("No results found in results/07_temporal_oos/. Run 07_temporal_oos.py first.")
            sys.exit(1)
        result_dir = dirs[-1]
    print(f"Loading results from {result_dir}")

    json_path = result_dir / "temporal_oos_results.json"
    with open(json_path) as f:
        data = json.load(f)

    meta    = data["metadata"]
    results = data["results"]
    train_date = meta["train_date"]
    test_date  = meta["test_date"]
    N, R_N     = meta["n_factors"], meta["r_n"]
    seed       = meta["seed"]

    # ── Load surfaces ─────────────────────────────────────────────────────────
    print(f"Loading {train_date} surface ...")
    surf_tr, fv_tr, S0_tr = _load_surface(train_date, args.data_dir)
    mkt_tr = surf_tr.ivs_per_T()
    K_tr   = surf_tr.strikes_per_T()
    F_tr   = surf_tr.forward_per_T()

    print(f"Loading {test_date}  surface ...")
    surf_te, fv_te, S0_te = _load_surface(test_date, args.data_dir)
    mkt_te = surf_te.ivs_per_T()
    K_te   = surf_te.strikes_per_T()
    F_te   = surf_te.forward_per_T()

    # ── Reprice all models ────────────────────────────────────────────────────
    model_tr, model_te = {}, {}
    kernel_map = {
        "LH-geo": ("lh", "geometric"),
        "LH-L2":  ("lh", "l2"),
        "aB-L2":  ("ab", "l2"),
        "aB-geo": ("ab", "geometric"),
    }

    for name, rec in results.items():
        p = rec["params"]
        family, kernel = kernel_map.get(name, (None, None))
        if family is None:
            continue
        if args.no_ab and family == "ab":
            print(f"  Skipping {name} (--no-ab)")
            continue

        print(f"  Repricing {name} on {train_date} ...")
        if family == "lh":
            model_tr[name] = _reprice_lh(p, kernel, N, R_N, fv_tr, S0_tr, K_tr)
        else:
            model_tr[name] = _reprice_ab(p, kernel, N, R_N, fv_tr, S0_tr, K_tr,
                                          args.M_paths, seed + 10)

        print(f"  Repricing {name} on {test_date}  ...")
        if family == "lh":
            model_te[name] = _reprice_lh(p, kernel, N, R_N, fv_te, S0_te, K_te)
        else:
            model_te[name] = _reprice_ab(p, kernel, N, R_N, fv_te, S0_te, K_te,
                                          args.M_paths, seed + 20)

    # ── Figure 1 — IV slices on train date ───────────────────────────────────
    T_plot_tr = _pick_maturities(list(mkt_tr.keys()), n=6)
    fig1 = _iv_slice_panel(
        f"IV slices — train date {train_date} (IS)",
        mkt_tr, K_tr, F_tr, model_tr, S0_tr, T_plot_tr,
    )
    p1 = result_dir / "iv_slices_train.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved {p1}")
    plt.close(fig1)

    # ── Figure 2 — IV slices on test date ────────────────────────────────────
    T_plot_te = _pick_maturities(list(mkt_te.keys()), n=6)
    fig2 = _iv_slice_panel(
        f"IV slices — test date {test_date} (temporal OOS)",
        mkt_te, K_te, F_te, model_te, S0_te, T_plot_te,
    )
    p2 = result_dir / "iv_slices_test.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved {p2}")
    plt.close(fig2)

    # ── Figure 3 — ATM skew on both dates ────────────────────────────────────
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False,
                              constrained_layout=True)
    fig3.suptitle(r"ATM skew $|\partial\sigma/\partial k|_{k=0}|$", fontsize=12)

    for ax, date, mkt_ivs, mkt_K, mkt_F, model_ivs in [
        (axes[0], train_date, mkt_tr, K_tr, F_tr, model_tr),
        (axes[1], test_date,  mkt_te, K_te, F_te, model_te),
    ]:
        T_mkt, sk_mkt = _atm_skew(mkt_ivs, mkt_K, mkt_F)
        ax.plot(T_mkt, sk_mkt, **MKT_STYLE)
        for name, ivs_d in model_ivs.items():
            T_m, sk_m = _atm_skew(ivs_d, mkt_K, mkt_F)
            style = STYLES.get(name, {})
            ax.plot(T_m, sk_m, **style)
        ax.set_xscale("log")
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel(r"$|\partial \sigma_{\rm impl}/\partial k|_{k=0}|$")
        label = "IS" if date == train_date else "Temporal OOS"
        ax.set_title(f"{date} ({label})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    p3 = result_dir / "atm_skew.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Saved {p3}")
    plt.close(fig3)

    # ── Figure 4 — IS vs OOS bar chart ───────────────────────────────────────
    names = [n for n in results if n in model_tr]
    x = np.arange(len(names))
    w = 0.2

    fig4, (ax_iv, ax_sk) = plt.subplots(1, 2, figsize=(11, 4.5),
                                          constrained_layout=True)
    fig4.suptitle(f"IS vs Temporal OOS  ({train_date} → {test_date})", fontsize=12)

    colors_is  = ["steelblue", "dodgerblue", "darkorange", "saddlebrown"]
    colors_oos = ["lightsteelblue", "lightskyblue", "bisque", "burlywood"]

    for ax, metric, ylabel in [
        (ax_iv, "iv_rmse",   "IV RMSE"),
        (ax_sk, "skew_rmse", "Skew RMSE"),
    ]:
        is_vals  = [results[n]["in_sample"][metric]   for n in names]
        oos_vals = [results[n]["temporal_oos"][metric] for n in names]

        for i, (ci, co) in enumerate(zip(colors_is[:len(names)],
                                          colors_oos[:len(names)])):
            ax.bar(x[i] - w/2, is_vals[i],  width=w, color=ci,
                   label="IS"  if i == 0 else "_")
            ax.bar(x[i] + w/2, oos_vals[i], width=w, color=co,
                   label="OOS" if i == 0 else "_", edgecolor="grey", lw=0.5)
            ax.text(x[i] - w/2, is_vals[i]  + 0.001, f"{is_vals[i]:.3f}",
                    ha="center", va="bottom", fontsize=7)
            ax.text(x[i] + w/2, oos_vals[i] + 0.001, f"{oos_vals[i]:.3f}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    p4 = result_dir / "is_vs_oos_bars.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    print(f"Saved {p4}")
    plt.close(fig4)

    print(f"\nAll figures saved to {result_dir}")
    print("  iv_slices_train.png  — IS fit quality")
    print("  iv_slices_test.png   — Temporal OOS fit quality")
    print("  atm_skew.png         — ATM skew both dates side by side")
    print("  is_vs_oos_bars.png   — IV-RMSE and Skew-RMSE comparison bars")


if __name__ == "__main__":
    main()
