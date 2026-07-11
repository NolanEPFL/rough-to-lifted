"""scripts/01_lh_convergence.py — Experiment 1.

Lifted Heston convergence in n. For H ∈ {0.05, 0.10, 0.20, 0.30} and
n ∈ {2, 4, 6, 10, 20, 50}, compute the IV surface and compare to the
reference n=500 surface.  Geometric grid requires even n.

Fixed parameters: (ν, ρ) = (0.4, −0.7), flat ξ₀ = 0.04.

Outputs
-------
results/01_lh_convergence/<run_id>/
    convergence_metrics.json
    convergence_plot.pdf
    params.json
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.common.forward_variance import FlatForwardVariance

# Phase 4: rough Heston reference (optional)
try:
    from src.rough_heston.pricing import rough_heston_iv_surface
    _ROUGH_HESTON_AVAILABLE = True
except Exception:
    _ROUGH_HESTON_AVAILABLE = False


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/01_lh_convergence"))
    parser.add_argument("--use-rough-heston-reference", action="store_true",
                        help="Phase 4: use rough Heston as reference (slower; else n=500 LH).")
    args = parser.parse_args()

    run_dir = _run_dir(args.out, args.seed)
    _setup_mpl()

    H_vals       = [0.05, 0.10, 0.20, 0.30]
    n_vals       = [2, 4, 6, 10, 20, 50]    # even (geometric grid requirement)
    n_ref        = 500
    nu, rho      = 0.4, -0.7
    V0           = 0.04
    S0           = 100.0
    use_rough    = args.use_rough_heston_reference and _ROUGH_HESTON_AVAILABLE
    rough_n_steps = 500    # fast reference; use 1000 for publication

    fv            = FlatForwardVariance(V0)
    maturities    = [0.1, 0.25, 0.5, 1.0, 2.0]
    strikes       = S0 * np.exp(np.linspace(-0.4, 0.2, 13))
    strikes_per_T = {T: strikes for T in maturities}

    hyperparams = {
        "H_vals": H_vals, "n_vals": n_vals, "n_ref": n_ref,
        "use_rough_heston": use_rough,
        "nu": nu, "rho": rho, "V0": V0, "S0": S0,
        "maturities": maturities,
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    all_metrics: dict = {}
    print(f"Running Experiment 1 → {run_dir}")
    if use_rough:
        print("  Using rough Heston as reference (Phase 4).")

    for H in H_vals:
        if use_rough:
            print(f"  H = {H:.2f}  building rough Heston reference ...", flush=True)
            ref_ivs = rough_heston_iv_surface(
                H=H, nu=nu, rho=rho,
                forward_variance=fv, S0=S0,
                strikes_per_T=strikes_per_T,
                n_steps=rough_n_steps,
            )
        else:
            print(f"  H = {H:.2f}  building reference (n={n_ref}) ...", flush=True)
            ref_params = LiftedHestonParams(H=H, n=n_ref, r_n=2.5, nu=nu, rho=rho)
            ref_ivs    = lifted_heston_iv_surface(ref_params, fv, S0, strikes_per_T)

        metrics_H: dict = {}
        for n in n_vals:
            print(f"    n = {n:3d} ...", end=" ", flush=True)
            params = LiftedHestonParams(H=H, n=n, r_n=2.5, nu=nu, rho=rho)
            ivs    = lifted_heston_iv_surface(params, fv, S0, strikes_per_T)

            sup_err = 0.0
            sq_tot  = 0.0
            count   = 0
            for T in maturities:
                valid = np.isfinite(ivs[T]) & np.isfinite(ref_ivs[T])
                diff  = np.abs(ivs[T][valid] - ref_ivs[T][valid])
                if diff.size:
                    sup_err  = max(sup_err, float(diff.max()))
                    sq_tot  += float((diff ** 2).sum())
                    count   += int(valid.sum())

            rmse = float(np.sqrt(sq_tot / count)) if count else np.nan
            metrics_H[n] = {"sup": sup_err, "rmse": rmse}
            print(f"sup={sup_err:.2e}  rmse={rmse:.2e}")

        all_metrics[H] = metrics_H

    # Save JSON
    with open(run_dir / "convergence_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Plot log-log error vs n
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(H_vals)))
    for ax, metric in zip(axes, ["sup", "rmse"]):
        for H, col in zip(H_vals, colors):
            ns   = list(all_metrics[H].keys())
            vals = [all_metrics[H][n][metric] for n in ns]
            ax.loglog(ns, vals, "o-", color=col, label=f"H={H}", ms=4)
        ax.set_xlabel("n")
        ax.set_ylabel("sup|Δσ|" if metric == "sup" else "RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "convergence_plot.pdf", bbox_inches="tight")
    plt.close()
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
