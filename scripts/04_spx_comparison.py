"""scripts/04_spx_comparison.py — Experiment 4. **MAIN EMPIRICAL RESULT.**

Three-model comparison on one cleaned SPX date (or synthetic surface):
  LH-geo  — Lifted Heston, geometric kernel, n = 20.
  aB-L²   — aBergomi, L²-fit kernel, n = 20.
  aB-geo  — aBergomi, geometric kernel (same as LH-geo).

Each model is calibrated in-sample (T ≤ in_sample_T_max) and evaluated
out-of-sample (T > in_sample_T_max).

Real data (--data CSV): loaded via src.data.spx_loader.load_spx_csv.
Synthetic fallback (--synthetic): uses make_synthetic_surface.

Outputs
-------
results/04_spx/<run_id>/
    comparison_table.tex
    calibrated_params.json
    iv_slices.pdf
    atm_skew.pdf
    residuals.pdf
    diagnostics.txt
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import load_spx_csv, make_synthetic_surface, fit_xi0_from_surface
from src.common.forward_variance import FlatForwardVariance, PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.calibration.optimizer import calibrate_lifted_heston, calibrate_abergomi
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.kernel_fit import ABergomiParams
from src.abergomi.pricing import abergomi_iv_surface


def _thin_surface(
    market_ivs: dict, strikes_per: dict,
    max_mats: int = 8, max_strikes: int = 15,
) -> tuple[dict, dict]:
    """Subsample to at most max_mats log-spaced maturities and max_strikes per maturity."""
    all_T = np.array(sorted(market_ivs.keys()))
    if len(all_T) > max_mats:
        # Pick maturities closest to log-spaced targets spanning [T_min, T_max]
        targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), max_mats))
        idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
        idx = sorted(set(idx))
        all_T = all_T[idx]

    new_ivs, new_K = {}, {}
    for T in all_T:
        ivs = market_ivs[T]
        Ks  = strikes_per[T]
        if len(Ks) > max_strikes:
            idx = np.round(np.linspace(0, len(Ks) - 1, max_strikes)).astype(int)
            Ks  = Ks[idx]
            ivs = ivs[idx]
        new_ivs[float(T)] = ivs
        new_K[float(T)]   = Ks
    return new_ivs, new_K


def _run_dir(out_root: Path, label: str, seed: int) -> Path:
    try:
        gh = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=str(_ROOT),
        ).decode().strip()
    except Exception:
        gh = "nogit"
    d = out_root / f"{label}_{gh}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_mpl():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "figure.figsize": (5.5, 3.5),
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--data",            type=Path,  default=None,
                        help="Path to cleaned SPX CSV (omit for synthetic mode).")
    parser.add_argument("--synthetic",       action="store_true",
                        help="Use synthetic surface instead of real data.")
    parser.add_argument("--quote-date",      type=str,   default="synthetic",
                        help="Date label for the run_id.")
    parser.add_argument("--in-sample-T-max", type=float, default=1,
                        help="IS/OOS split in years (default 1 = 1 year).")
    parser.add_argument("--n-factors",       type=int,   default=20)
    parser.add_argument("--r-n",             type=float, default=2.5)
    parser.add_argument("--M-paths",         type=int,   default=20_000)
    parser.add_argument("--de-maxiter",      type=int,   default=15,
                        help="DE iterations. Use 30 for production quality.")
    parser.add_argument("--lh-l2",           action="store_true",
                        help="Phase 4: also calibrate LH with L²-fit kernel (4th column).")
    parser.add_argument("--max-maturities",  type=int,   default=8,
                        help="Max maturities to keep after thinning (log-spaced).")
    parser.add_argument("--max-strikes",     type=int,   default=15,
                        help="Max strikes per maturity to keep after thinning.")
    parser.add_argument("--out",             type=Path,  default=Path("results/04_spx"))
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        print("No --data provided; running in synthetic mode (--synthetic).")
        args.synthetic = True

    run_dir = _run_dir(args.out, args.quote_date, args.seed)
    _setup_mpl()

    # ── 1. Load or generate surface ─────────────────────────────────────────
    S0 = 100.0
    if args.synthetic:
        print("Generating synthetic market surface ...")
        surface = make_synthetic_surface(seed=args.seed, noise_std=0.001)
        fv_flat = FlatForwardVariance(0.04)
        forward_variance = fv_flat
    else:
        print(f"Loading {args.data} ...")
        surface = load_spx_csv(args.data)
        # Build ξ₀ from ATM total variance
        xi0_mats, xi0_vals = fit_xi0_from_surface(surface)
        forward_variance = PiecewiseConstantForwardVariance(xi0_mats, xi0_vals)
        S0 = float(surface.forwards.mean())

    market_ivs  = surface.ivs_per_T()
    strikes_per = surface.strikes_per_T()

    # Thin the surface to at most max_mats maturities and max_strikes per
    # maturity so DE calibration stays tractable on real data.
    market_ivs, strikes_per = _thin_surface(
        market_ivs, strikes_per,
        max_mats=args.max_maturities, max_strikes=args.max_strikes,
    )
    all_T = sorted(market_ivs.keys())
    print(f"Surface after thinning: {len(all_T)} maturities, "
          f"{sum(len(v) for v in market_ivs.values())} points total")

    loss = IVSurfaceLoss(
        market_ivs=market_ivs, strikes=strikes_per, S0=S0,
        forwards=surface.forward_per_T(),
        weights="vega",
        skew_lambda=0.0,
        in_sample_T_max=args.in_sample_T_max,
    )

    with open(run_dir / "params.json", "w") as f:
        json.dump({
            "seed": args.seed, "synthetic": args.synthetic,
            "in_sample_T_max": args.in_sample_T_max,
            "n_factors": args.n_factors, "M_paths": args.M_paths,
        }, f, indent=2)

    calib_all: dict = {}
    ivs_models: dict[str, dict] = {}

    common_lh = dict(
        n=args.n_factors, r_n=args.r_n, seed=args.seed,
        de_popsize=10, de_maxiter=args.de_maxiter, refine=True,
    )
    common_ab = dict(
        n=args.n_factors, r_n=args.r_n, seed=args.seed,
        M_paths=args.M_paths, de_popsize=10, de_maxiter=args.de_maxiter,
        refine=True, final_M_paths=min(args.M_paths * 4, 100_000),
    )

    # ── 2. LH-geo ────────────────────────────────────────────────────────────
    print("\n[1/4] Calibrating LH-geo ...")
    res_lh = calibrate_lifted_heston(loss, forward_variance, **common_lh)
    print(f"      IS={res_lh.loss_in_sample:.4f}  OOS={res_lh.loss_out_of_sample}  "
          f"t={res_lh.elapsed_seconds:.0f}s  {res_lh.params}")
    calib_all["LH-geo"] = {
        "params": res_lh.params, "IS_RMSE": res_lh.loss_in_sample,
        "OOS_RMSE": res_lh.loss_out_of_sample, "n_evals": res_lh.n_evals,
        "elapsed_s": res_lh.elapsed_seconds,
    }
    p = res_lh.params
    lh_geo_params = LiftedHestonParams(H=p["H"], n=args.n_factors, r_n=args.r_n,
                                        nu=p["nu"], rho=p["rho"])
    ivs_models["LH-geo"] = lifted_heston_iv_surface(
        lh_geo_params, forward_variance, S0, strikes_per)

    # ── 3. Phase 4 optional: LH-L² (runs before aBergomi — fast check) ───────
    if args.lh_l2:
        print("[2/4] Calibrating LH-L² ...")
        res_lh_l2 = calibrate_lifted_heston(
            loss, forward_variance, kernel="l2", **common_lh)
        print(f"      IS={res_lh_l2.loss_in_sample:.4f}  "
              f"OOS={res_lh_l2.loss_out_of_sample}  "
              f"t={res_lh_l2.elapsed_seconds:.0f}s  {res_lh_l2.params}")
        calib_all["LH-L2"] = {
            "params": res_lh_l2.params, "IS_RMSE": res_lh_l2.loss_in_sample,
            "OOS_RMSE": res_lh_l2.loss_out_of_sample, "n_evals": res_lh_l2.n_evals,
            "elapsed_s": res_lh_l2.elapsed_seconds,
        }
        p_lhl2 = res_lh_l2.params
        lh_l2_params = LiftedHestonParams(
            H=p_lhl2["H"], n=args.n_factors, r_n=args.r_n,
            nu=p_lhl2["nu"], rho=p_lhl2["rho"], kernel="l2",
        )
        ivs_models["LH-L2"] = lifted_heston_iv_surface(
            lh_l2_params, forward_variance, S0, strikes_per)

    # ── 4. aB-L² ─────────────────────────────────────────────────────────────
    n_total = 4 if args.lh_l2 else 3
    print(f"[{n_total - 1}/{n_total}] Calibrating aB-L² ...")
    res_ab_l2 = calibrate_abergomi(loss, forward_variance, kernel="l2", **common_ab)
    print(f"      IS={res_ab_l2.loss_in_sample:.4f}  OOS={res_ab_l2.loss_out_of_sample}  "
          f"t={res_ab_l2.elapsed_seconds:.0f}s  {res_ab_l2.params}")
    calib_all["aB-L2"] = {
        "params": res_ab_l2.params, "IS_RMSE": res_ab_l2.loss_in_sample,
        "OOS_RMSE": res_ab_l2.loss_out_of_sample, "n_evals": res_ab_l2.n_evals,
        "elapsed_s": res_ab_l2.elapsed_seconds,
    }
    p2 = res_ab_l2.params
    ab_l2_params = ABergomiParams(H=p2["H"], n=args.n_factors,
                                   eta=p2["eta"], rho=p2["rho"], kernel="l2")
    ivs_models["aB-L2"], _ = abergomi_iv_surface(
        ab_l2_params, forward_variance, S0, strikes_per,
        M_paths=min(args.M_paths * 4, 100_000), qmc=False, seed=args.seed + 1,
    )

    # ── 5. aB-geo ─────────────────────────────────────────────────────────────
    print(f"[{n_total}/{n_total}] Calibrating aB-geo ...")
    res_ab_geo = calibrate_abergomi(loss, forward_variance, kernel="geometric", **common_ab)
    print(f"      IS={res_ab_geo.loss_in_sample:.4f}  OOS={res_ab_geo.loss_out_of_sample}  "
          f"t={res_ab_geo.elapsed_seconds:.0f}s  {res_ab_geo.params}")
    calib_all["aB-geo"] = {
        "params": res_ab_geo.params, "IS_RMSE": res_ab_geo.loss_in_sample,
        "OOS_RMSE": res_ab_geo.loss_out_of_sample, "n_evals": res_ab_geo.n_evals,
        "elapsed_s": res_ab_geo.elapsed_seconds,
    }
    p3 = res_ab_geo.params
    ab_geo_params = ABergomiParams(H=p3["H"], n=args.n_factors,
                                    eta=p3["eta"], rho=p3["rho"], kernel="geometric",
                                    r_n=args.r_n)
    ivs_models["aB-geo"], _ = abergomi_iv_surface(
        ab_geo_params, forward_variance, S0, strikes_per,
        M_paths=min(args.M_paths * 4, 100_000), qmc=False, seed=args.seed + 2,
    )

    # ── 6. Save JSON ─────────────────────────────────────────────────────────
    with open(run_dir / "calibrated_params.json", "w") as f:
        json.dump(calib_all, f, indent=2)

    # ── 6. LaTeX comparison table ─────────────────────────────────────────────
    tex = _comparison_table(calib_all)
    with open(run_dir / "comparison_table.tex", "w") as f:
        f.write(tex)

    # ── 7. Figures ────────────────────────────────────────────────────────────
    # Pick 5 log-spaced maturities from the actual thinned surface
    _all_T = np.array(sorted(market_ivs.keys()))
    _idx   = np.round(np.linspace(0, len(_all_T) - 1, min(5, len(_all_T)))).astype(int)
    plot_mats = [float(_all_T[i]) for i in _idx]

    _plot_iv_slices(S0, strikes_per, market_ivs, ivs_models,
                    plot_mats, run_dir / "iv_slices.pdf")
    _plot_atm_skew(S0, strikes_per, market_ivs, ivs_models,
                   run_dir / "atm_skew.pdf")
    _plot_residuals(S0, strikes_per, market_ivs, ivs_models,
                    run_dir / "residuals.pdf")

    # ── 8. Diagnostics text ───────────────────────────────────────────────────
    diag = [f"Run: {run_dir}", f"Date: {surface.date}", ""]
    for name, res in calib_all.items():
        diag.append(f"{name}: {res}")
    with open(run_dir / "diagnostics.txt", "w") as f:
        f.write("\n".join(diag))

    print(f"\nAll done. Results in {run_dir}")


def _comparison_table(calib_all: dict) -> str:
    hdr = (r"\begin{tabular}{lcccccc}" + "\n"
           r"\hline" + "\n"
           r"Model & H & param & $\rho$ & IS RMSE & OOS RMSE & Time (s) \\" + "\n"
           r"\hline")
    rows = [hdr]
    names = {"LH-geo": ("$\\nu$", "nu"), "LH-L2": ("$\\nu$", "nu"),
             "aB-L2": ("$\\eta$", "eta"), "aB-geo": ("$\\eta$", "eta")}
    for name, res in calib_all.items():
        p    = res["params"]
        H    = p.get("H", float("nan"))
        rho  = p.get("rho", float("nan"))
        lbl, key = names.get(name, ("?", "?"))
        val  = p.get(key, float("nan"))
        IS   = res.get("IS_RMSE", float("nan"))
        OOS  = res.get("OOS_RMSE") or float("nan")
        t    = res.get("elapsed_s", float("nan"))
        rows.append(
            f"{name} & {H:.3f} & {lbl}$={val:.3f}$ & {rho:.3f} & "
            f"{IS:.4f} & {OOS:.4f} & {t:.0f} \\\\"
        )
    rows += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(rows)


def _plot_iv_slices(S0, strikes_per_T, mkt_ivs, ivs_models, mats, path):
    n_m   = len(mats)
    cols  = min(n_m, 3)
    rows_ = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(5.5, 2.2 * rows_), squeeze=False)
    styles = {"LH-geo": ("b-", 1.2), "LH-L2": ("b--", 1.0), "aB-L2": ("r--", 1.2), "aB-geo": ("g:", 1.2)}
    for idx, T in enumerate(mats):
        ax  = axes[idx // cols][idx % cols]
        K   = strikes_per_T.get(T, np.array([]))
        k   = np.log(K / S0)
        if T in mkt_ivs:
            ax.plot(k, mkt_ivs[T], "k.", ms=4, label="Mkt", zorder=5)
        for name, (sty, lw) in styles.items():
            if T in ivs_models.get(name, {}):
                ax.plot(k, ivs_models[name][T], sty, lw=lw, label=name)
        ax.set_title(f"T={T:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)
    for idx in range(n_m, rows_ * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_atm_skew(S0, strikes_per_T, mkt_ivs, ivs_models, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    def _skew_ts(ivs_dict):
        ms, ss = [], []
        for T in sorted(ivs_dict):
            K  = strikes_per_T.get(T, np.array([]))
            iv = ivs_dict[T]
            if len(K) == 0 or len(K) != len(iv):
                continue
            k  = np.log(K / S0)
            fin = np.isfinite(iv) & np.isfinite(k)
            if fin.sum() < 2:
                continue
            idx = np.argsort(np.abs(k[fin]))[:2]
            k_f, iv_f = k[fin][idx], iv[fin][idx]
            dk = k_f[1] - k_f[0]
            if abs(dk) < 1e-8:
                continue
            ms.append(T); ss.append((iv_f[1] - iv_f[0]) / dk)
        return np.array(ms), np.array(ss)

    m, s = _skew_ts(mkt_ivs)
    if m.size:
        ax.plot(m, s, "k.", ms=5, label="Market", zorder=5)

    styles = {"LH-geo": "b-", "LH-L2": "b--", "aB-L2": "r--", "aB-geo": "g:"}
    for name, sty in styles.items():
        if name in ivs_models:
            m2, s2 = _skew_ts(ivs_models[name])
            if m2.size:
                ax.plot(m2, s2, sty, lw=1.2, label=name)

    ax.set_xlabel("T (years)"); ax.set_ylabel("ATM skew")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_residuals(S0, strikes_per_T, mkt_ivs, ivs_models, path):
    """Residual heatmap (model - market) on (k, T) per model."""
    n_models = len(ivs_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5, 2.5), squeeze=False)
    for ax, (name, mdl_ivs) in zip(axes.flat, ivs_models.items()):
        k_all, T_all, res_all = [], [], []
        for T in sorted(mkt_ivs):
            if T not in mdl_ivs:
                continue
            K    = strikes_per_T.get(T, np.array([]))
            mkt  = mkt_ivs[T]
            mdl  = mdl_ivs[T]
            fin  = np.isfinite(mkt) & np.isfinite(mdl)
            k_all.extend(np.log(K[fin] / S0).tolist())
            T_all.extend([T] * fin.sum())
            res_all.extend((mdl[fin] - mkt[fin]).tolist())
        if not k_all:
            ax.set_visible(False)
            continue
        k_arr  = np.array(k_all)
        T_arr  = np.array(T_all)
        res_arr = np.array(res_all)
        sc = ax.scatter(k_arr, T_arr, c=res_arr, cmap="RdBu_r",
                        vmin=-0.05, vmax=0.05, s=12)
        plt.colorbar(sc, ax=ax, label="model−mkt IV")
        ax.set_xlabel("k"); ax.set_ylabel("T")
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
