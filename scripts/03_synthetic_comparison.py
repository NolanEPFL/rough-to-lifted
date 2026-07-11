"""scripts/03_synthetic_comparison.py — Experiment 3.

Synthetic comparison: generate a "market" surface from n=500 lifted Heston
(known ground truth), then calibrate LH-geo (n=20) and aB-L² (n=20) to it.

Purpose: verify both models can recover a ground truth they approximate, and
quantify calibration bias.

Outputs
-------
results/03_synthetic/<run_id>/
    calibrated_params.json
    summary.tex
    iv_slices.pdf
    atm_skew.pdf
    params.json
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spx_loader import make_synthetic_surface
from src.common.forward_variance import FlatForwardVariance, PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.calibration.optimizer import calibrate_lifted_heston, calibrate_abergomi
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.kernel_fit import ABergomiParams
from src.abergomi.pricing import abergomi_iv_surface


def _run_dir(out_root: Path, seed: int) -> Path:
    try:
        gh = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=str(_ROOT),
        ).decode().strip()
    except Exception:
        gh = "nogit"
    d = out_root / f"{gh}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_mpl():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "figure.figsize": (5.5, 3.5),
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _atm_skew_ts(S0, strikes_per_T, ivs_per_T):
    """ATM skew term structure from an IV surface dict."""
    mats, skews = [], []
    for T in sorted(ivs_per_T.keys()):
        K = strikes_per_T.get(T)
        iv = ivs_per_T.get(T)
        if K is None or iv is None:
            continue
        k = np.log(K / S0)
        finite = np.isfinite(iv)
        k_f, iv_f = k[finite], iv[finite]
        if len(k_f) < 2:
            continue
        idx = np.argsort(np.abs(k_f))[:2]
        dk  = k_f[idx[1]] - k_f[idx[0]]
        if abs(dk) < 1e-8:
            continue
        mats.append(T)
        skews.append((iv_f[idx[1]] - iv_f[idx[0]]) / dk)
    return np.array(mats), np.array(skews)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--H-true",    type=float, default=0.10)
    parser.add_argument("--nu-true",   type=float, default=0.40)
    parser.add_argument("--rho-true",  type=float, default=-0.70)
    parser.add_argument("--V0",        type=float, default=0.04)
    parser.add_argument("--n-factors", type=int,   default=20)
    parser.add_argument("--M-paths",   type=int,   default=20_000)
    parser.add_argument("--de-maxiter",type=int,   default=15,
                        help="DE iterations (use 30 for production)")
    parser.add_argument("--out",       type=Path,  default=Path("results/03_synthetic"))
    args = parser.parse_args()

    run_dir = _run_dir(args.out, args.seed)
    _setup_mpl()

    S0  = 100.0
    maturities = [0.083, 0.25, 0.5, 1.0, 1.5, 2.0]
    fv  = FlatForwardVariance(args.V0)

    # ── 1. Generate synthetic "market" ─────────────────────────────────────
    print("Generating synthetic market surface (n=500 LH) ...")
    surface = make_synthetic_surface(
        H=args.H_true, nu=args.nu_true, rho=args.rho_true, V0=args.V0,
        S0=S0, maturities=maturities, n_strikes=13, seed=args.seed,
        noise_std=0.001,
    )

    market_ivs  = surface.ivs_per_T()
    strikes_per = surface.strikes_per_T()

    loss = IVSurfaceLoss(
        market_ivs=market_ivs, strikes=strikes_per, S0=S0,
        weights="equal", in_sample_T_max=1.0,
    )

    hyperparams = vars(args)
    hyperparams["out"] = str(args.out)
    with open(run_dir / "params.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    calib_results: dict = {}

    # ── 2. Calibrate LH-geo ─────────────────────────────────────────────────
    print("\nCalibrating LH-geo ...")
    t0 = time.perf_counter()
    res_lh = calibrate_lifted_heston(
        loss, fv, n=args.n_factors, r_n=2.5, seed=args.seed,
        de_popsize=10, de_maxiter=args.de_maxiter, refine=True,
    )
    print(f"  Elapsed: {res_lh.elapsed_seconds:.1f}s  "
          f"IS-RMSE: {res_lh.loss_in_sample:.4f}  "
          f"OOS-RMSE: {res_lh.loss_out_of_sample:.4f}  "
          f"params: {res_lh.params}")
    calib_results["LH-geo"] = {
        "params": res_lh.params,
        "IS_RMSE": res_lh.loss_in_sample,
        "OOS_RMSE": res_lh.loss_out_of_sample,
        "n_evals": res_lh.n_evals,
        "elapsed_s": res_lh.elapsed_seconds,
    }

    # ── 3. Calibrate aB-L² ──────────────────────────────────────────────────
    print("\nCalibrating aB-L² ...")
    res_ab = calibrate_abergomi(
        loss, fv, n=args.n_factors, kernel="l2", seed=args.seed,
        M_paths=args.M_paths, de_popsize=10, de_maxiter=args.de_maxiter,
        refine=True, final_M_paths=min(args.M_paths * 4, 100_000),
    )
    print(f"  Elapsed: {res_ab.elapsed_seconds:.1f}s  "
          f"IS-RMSE: {res_ab.loss_in_sample:.4f}  "
          f"OOS-RMSE: {res_ab.loss_out_of_sample:.4f}  "
          f"params: {res_ab.params}")
    calib_results["aB-L2"] = {
        "params": res_ab.params,
        "IS_RMSE": res_ab.loss_in_sample,
        "OOS_RMSE": res_ab.loss_out_of_sample,
        "n_evals": res_ab.n_evals,
        "elapsed_s": res_ab.elapsed_seconds,
    }

    with open(run_dir / "calibrated_params.json", "w") as f:
        json.dump(calib_results, f, indent=2)

    # ── 4. Build model surfaces for plotting ────────────────────────────────
    p_lh = res_lh.params
    params_lh = LiftedHestonParams(
        H=p_lh["H"], n=args.n_factors, r_n=2.5,
        nu=p_lh["nu"], rho=p_lh["rho"],
    )
    ivs_lh = lifted_heston_iv_surface(params_lh, fv, S0, strikes_per)

    p_ab   = res_ab.params
    params_ab = ABergomiParams(H=p_ab["H"], n=args.n_factors,
                                eta=p_ab["eta"], rho=p_ab["rho"], kernel="l2")
    ivs_ab, _ = abergomi_iv_surface(
        params_ab, fv, S0, strikes_per,
        M_paths=args.M_paths * 4, qmc=False, seed=args.seed + 1,
    )

    # ── 5. LaTeX summary table ──────────────────────────────────────────────
    true_params = {"H": args.H_true, "nu": args.nu_true, "rho": args.rho_true}
    tex = _make_tex_table(calib_results, true_params)
    with open(run_dir / "summary.tex", "w") as f:
        f.write(tex)

    # ── 6. IV-slice plots ───────────────────────────────────────────────────
    plot_mats = [T for T in sorted(market_ivs.keys()) if T in [0.083, 0.25, 0.5, 1.0, 2.0]][:5]
    _plot_iv_slices(S0, strikes_per, market_ivs, ivs_lh, ivs_ab,
                    plot_mats, run_dir / "iv_slices.pdf")

    # ── 7. ATM skew plot ────────────────────────────────────────────────────
    _plot_atm_skew(S0, strikes_per, market_ivs, ivs_lh, ivs_ab,
                   run_dir / "atm_skew.pdf",
                   H_true=args.H_true)

    print(f"\nDone. Results in {run_dir}")


def _make_tex_table(results: dict, true_params: dict) -> str:
    rows = []
    rows.append(r"\begin{tabular}{lrrrrrr}")
    rows.append(r"\hline")
    rows.append(r"Model & H & $\nu/\eta$ & $\rho$ & IS RMSE & OOS RMSE & Evals \\")
    rows.append(r"\hline")
    rows.append(
        f"True & {true_params['H']:.2f} & {true_params['nu']:.2f} & "
        f"{true_params['rho']:.2f} & --- & --- & --- \\\\"
    )
    for name, res in results.items():
        p = res["params"]
        H = p.get("H", float("nan"))
        main_param = p.get("nu", p.get("eta", float("nan")))
        rho = p.get("rho", float("nan"))
        rows.append(
            f"{name} & {H:.3f} & {main_param:.3f} & {rho:.3f} & "
            f"{res['IS_RMSE']:.4f} & {res.get('OOS_RMSE') or float('nan'):.4f} & "
            f"{res['n_evals']} \\\\"
        )
    rows.append(r"\hline")
    rows.append(r"\end{tabular}")
    return "\n".join(rows) + "\n"


def _plot_iv_slices(S0, strikes_per_T, mkt_ivs, ivs_lh, ivs_ab, mats, path):
    n_mats = len(mats)
    cols   = min(n_mats, 3)
    rows   = (n_mats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5, 2.0 * rows), squeeze=False)
    for idx, T in enumerate(mats):
        ax = axes[idx // cols][idx % cols]
        K  = strikes_per_T.get(T, np.array([]))
        k  = np.log(K / S0)
        if T in mkt_ivs:
            ax.plot(k, mkt_ivs[T], "k.", ms=4, label="Market")
        if T in ivs_lh:
            ax.plot(k, ivs_lh[T], "b-", lw=1.2, label="LH-geo")
        if T in ivs_ab:
            ax.plot(k, ivs_ab[T], "r--", lw=1.2, label="aB-L²")
        ax.set_title(f"T = {T:.2f}", fontsize=8)
        ax.set_xlabel("k", fontsize=8)
        ax.set_ylabel("IV", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(n_mats, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_atm_skew(S0, strikes_per_T, mkt_ivs, ivs_lh, ivs_ab, path, H_true=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for label, ivs_dict, style in [
        ("Market",  mkt_ivs, "k."), ("LH-geo", ivs_lh, "b-"), ("aB-L²", ivs_ab, "r--"),
    ]:
        mats_s, skews_s = [], []
        for T in sorted(ivs_dict.keys()):
            K  = strikes_per_T.get(T, np.array([]))
            iv = ivs_dict[T]
            if len(K) == 0 or len(K) != len(iv):
                continue
            k  = np.log(K / S0)
            fin = np.isfinite(iv) & np.isfinite(k)
            k_f, iv_f = k[fin], iv[fin]
            if len(k_f) < 2:
                continue
            idx = np.argsort(np.abs(k_f))[:2]
            dk  = k_f[idx[1]] - k_f[idx[0]]
            if abs(dk) < 1e-8:
                continue
            mats_s.append(T)
            skews_s.append((iv_f[idx[1]] - iv_f[idx[0]]) / dk)
        if mats_s:
            ax.plot(mats_s, skews_s, style, lw=1.2, label=label, ms=5)

    if H_true is not None:
        T_arr = np.linspace(0.05, 2.0, 50)
        rough = -0.5 * T_arr ** (H_true - 0.5)   # rough power-law reference (scaled)
        rough *= abs(ax.get_ylim()[0]) / abs(rough[0]) if ax.get_ylim()[0] != 0 else 1
        ax.plot(T_arr, rough, "g:", lw=1, label=f"rough T^(H-½), H={H_true}")

    ax.set_xlabel("T (years)")
    ax.set_ylabel("ATM skew")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
