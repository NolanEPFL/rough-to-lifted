"""scripts/05_make_thesis_figures.py — publication-quality figure compilation.

Reads JSON / CSV outputs from scripts 01–04 and regenerates figures with the
thesis-uniform matplotlib style. No new computation is performed.

Outputs
-------
results/thesis_figures/
    convergence.pdf
    kernel_errors.pdf
    synthetic_iv.pdf
    synthetic_skew.pdf
    spx_iv.pdf
    spx_skew.pdf
    spx_residuals.pdf  (if available)
"""

from __future__ import annotations

import sys
import argparse
import json
import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_mpl():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "figure.figsize": (5.5, 3.5),
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _latest_run(experiment_dir: Path) -> Path | None:
    """Return the most recently modified subdirectory under experiment_dir."""
    if not experiment_dir.exists():
        return None
    subs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return subs[0] if subs else None


def _fig1_convergence(run_dir: Path, out: Path) -> None:
    """Re-render convergence_plot from metrics JSON."""
    metrics_file = run_dir / "convergence_metrics.json"
    if not metrics_file.exists():
        print(f"  [skip] {metrics_file} not found")
        return

    with open(metrics_file) as f:
        data = json.load(f)

    H_vals = sorted(float(h) for h in data)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(H_vals)))

    for metric, ax in zip(["sup", "rmse"], axes):
        for H, col in zip(H_vals, colors):
            ns   = sorted(int(n) for n in data[str(H)])
            vals = [data[str(H)][str(n)][metric] for n in ns]
            ax.loglog(ns, vals, "o-", color=col, label=f"H={H}", ms=4, lw=1.2)
        ax.set_xlabel("$n$")
        ax.set_ylabel("$\\sup|\\Delta\\sigma|$" if metric == "sup" else "RMSE")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out / "convergence.pdf", bbox_inches="tight")
    plt.close()
    print(f"  convergence.pdf")


def _fig2_kernel_errors(run_dir: Path, out: Path) -> None:
    """Re-render kernel L² errors from CSV."""
    csv_file = run_dir / "kernel_errors.csv"
    if not csv_file.exists():
        print(f"  [skip] {csv_file} not found")
        return

    from collections import defaultdict
    data: dict = defaultdict(lambda: defaultdict(dict))
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[float(row["H"])][row["kernel"]][int(row["n"])] = float(row["L2"])

    H_vals = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors   = plt.cm.viridis(np.linspace(0.1, 0.9, len(H_vals)))
    styles   = {"geometric": "o-", "l2_fit": "s--"}

    for H, col in zip(H_vals, colors):
        for kernel, sty in styles.items():
            if kernel not in data[H]:
                continue
            d   = data[H][kernel]
            ns  = sorted(d.keys())
            l2s = [d[n] for n in ns]
            ax.semilogy(ns, l2s, sty, color=col, lw=1.2, ms=4,
                        label=f"H={H} {kernel}")

    ax.set_xlabel("$n$")
    ax.set_ylabel("$\\|K_H - K_n\\|_{L^2}$")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "kernel_errors.pdf", bbox_inches="tight")
    plt.close()
    print("  kernel_errors.pdf")


def _copy_figures(run_dir: Path, out: Path, prefix: str,
                  names: list[str]) -> None:
    """Copy or re-save PDFs from a run directory."""
    import shutil
    for name in names:
        src = run_dir / name
        if src.exists():
            dst = out / f"{prefix}_{name}"
            shutil.copy(src, dst)
            print(f"  {dst.name}")
        else:
            print(f"  [skip] {src} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out",          type=Path, default=Path("results/thesis_figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _setup_mpl()

    print("Compiling thesis figures ...")

    # Experiment 1
    r1 = _latest_run(args.results_root / "01_lh_convergence")
    if r1:
        print(f"Exp 1: {r1}")
        _fig1_convergence(r1, args.out)
    else:
        print("Exp 1: no results found")

    # Experiment 2
    r2 = _latest_run(args.results_root / "02_kernel_study")
    if r2:
        print(f"Exp 2: {r2}")
        _fig2_kernel_errors(r2, args.out)
        _copy_figures(r2, args.out, "kernel", ["kernel_plots.pdf"])
    else:
        print("Exp 2: no results found")

    # Experiment 3
    r3 = _latest_run(args.results_root / "03_synthetic")
    if r3:
        print(f"Exp 3: {r3}")
        _copy_figures(r3, args.out, "synthetic",
                      ["iv_slices.pdf", "atm_skew.pdf"])
    else:
        print("Exp 3: no results found")

    # Experiment 4
    r4 = _latest_run(args.results_root / "04_spx")
    if r4:
        print(f"Exp 4: {r4}")
        _copy_figures(r4, args.out, "spx",
                      ["iv_slices.pdf", "atm_skew.pdf", "residuals.pdf",
                       "comparison_table.tex"])
    else:
        print("Exp 4: no results found")

    print(f"Done. Figures in {args.out}")


if __name__ == "__main__":
    main()
