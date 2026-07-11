"""scripts/02_kernel_study.py — Experiment 2.

Kernel approximation error in isolation. For H ∈ {0.05, 0.10, 0.20, 0.30}
and n ∈ {4, 6, 10, 20}, compare geometric and L²-fit kernels:
    ‖K_H − K_n‖_{L²([ε, T_max])}  and  ‖K_H − K_n‖_∞([ε, T_max]).

This isolates whether the gap between LH-geo and aB-L² is a kernel artefact
or a model-dynamics effect.

Outputs
-------
results/02_kernel_study/<run_id>/
    kernel_errors.csv
    kernel_plots.pdf
    params.json
"""

from __future__ import annotations

import os
import sys
import argparse
import csv
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

from src.lifted_heston.params import geometric_grid
from src.lifted_heston.kernel import K_H, K_n, kernel_l2_error
from src.abergomi.kernel_fit import abergomi_l2_kernel


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


def _sup_norm_error(H: float, c: np.ndarray, x: np.ndarray,
                    eps: float, T_max: float, n_quad: int = 500) -> float:
    t = np.logspace(np.log10(eps), np.log10(T_max), n_quad)
    diff = np.abs(K_H(t, H) - K_n(t, c, x))
    return float(diff.max())


def _setup_mpl():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/02_kernel_study"))
    args = parser.parse_args()

    run_dir = _run_dir(args.out, args.seed)
    _setup_mpl()

    H_vals = [0.05, 0.10, 0.20, 0.30]
    n_vals = [4, 6, 10, 20]
    eps    = 1.0 / 365.0
    T_max  = 2.0

    hyperparams = {"H_vals": H_vals, "n_vals": n_vals, "eps": eps, "T_max": T_max}
    with open(run_dir / "params.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    rows = []
    print(f"Running Experiment 2 → {run_dir}")
    for H in H_vals:
        for n in n_vals:
            # Geometric kernel
            c_geo, x_geo = geometric_grid(H, n)
            l2_geo  = kernel_l2_error(H, c_geo, x_geo, eps=eps, T_max=T_max)
            sup_geo = _sup_norm_error(H, c_geo, x_geo, eps, T_max)

            # L²-fit kernel
            c_l2, x_l2, _ = abergomi_l2_kernel(H, n, eps=eps, T_max=T_max, seed=args.seed)
            l2_l2  = kernel_l2_error(H, c_l2, x_l2, eps=eps, T_max=T_max)
            sup_l2 = _sup_norm_error(H, c_l2, x_l2, eps, T_max)

            print(
                f"  H={H:.2f}  n={n:2d}  "
                f"geo  L2={l2_geo:.3e} sup={sup_geo:.3e}  |  "
                f"l2   L2={l2_l2:.3e} sup={sup_l2:.3e}"
            )
            rows.append({"H": H, "n": n, "kernel": "geometric",
                          "L2": l2_geo, "sup": sup_geo})
            rows.append({"H": H, "n": n, "kernel": "l2_fit",
                          "L2": l2_l2, "sup": sup_l2})

    # Save CSV
    with open(run_dir / "kernel_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["H", "n", "kernel", "L2", "sup"])
        writer.writeheader()
        writer.writerows(rows)

    # Plots: one panel per H, showing K_H and K_n for both kernels at n=20
    t_plot = np.logspace(np.log10(eps), np.log10(T_max), 300)
    n_show = 20
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
    for ax, H in zip(axes.flat, H_vals):
        ref = K_H(t_plot, H)
        ax.loglog(t_plot, ref, "k-", lw=1.5, label="K_H")

        c_geo, x_geo = geometric_grid(H, n_show)
        ax.loglog(t_plot, K_n(t_plot, c_geo, x_geo), "b--", lw=1, label="geo")

        c_l2, x_l2, _ = abergomi_l2_kernel(H, n_show, eps=eps, T_max=T_max,
                                             seed=args.seed)
        ax.loglog(t_plot, K_n(t_plot, c_l2, x_l2), "r:", lw=1, label="L²")

        ax.set_title(f"H = {H}", fontsize=8)
        ax.set_xlabel("t", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    plt.savefig(run_dir / "kernel_plots.pdf", bbox_inches="tight")
    plt.close()
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
