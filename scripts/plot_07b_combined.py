"""scripts/plot_07b_combined.py — Compare two-factor LH against single-factor models.

Loads saved params from both:
  results/07_temporal_oos/          (LH-geo, LH-L2, aB-L2, aB-geo)
  results/07b_temporal_oos_two_factor/   (TF-LH)

Produces, for each lambda in {0, inf}:
  1. iv_slices_train_lam{X}.png    — IV slices on train date, all models
  2. iv_slices_test_lam{X}.png     — IV slices on test date, all models
  3. atm_skew_lam{X}.png           — ATM skew both dates, all models
  4. tf_lh_only_lam{X}.png         — TF-LH vs market (IV slices, both dates)
  5. bars_all.png                  — IS vs OOS bars, all models, both lambdas

Usage
-----
    python scripts/plot_07b_combined.py
    python scripts/plot_07b_combined.py --train 2024-08-05 --test 2024-08-06
    python scripts/plot_07b_combined.py --no-ab    # skip slow aB MC repricing
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
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import two_factor_lh_iv_surface


# ── Styles ────────────────────────────────────────────────────────────────────
STYLES = {
    "LH-geo": dict(color="blue",   ls="-",  lw=1.5, label="LH-geo"),
    "LH-L2":  dict(color="blue",   ls="--", lw=1.2, label="LH-L²"),
    "aB-L2":  dict(color="red",    ls="--", lw=1.5, label="aB-L²"),
    "aB-geo": dict(color="green",  ls=":",  lw=1.5, label="aB-geo"),
    "TF-LH":  dict(color="purple", ls="-",  lw=2.0, label="TF-LH"),
}
MKT  = dict(color="black", marker="o", ms=3.5, lw=0, zorder=5, label="Market")
T_MIN = 1.0 / 52


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_surface(date, data_dir):
    surf = load_spx_csv(data_dir / f"spx_{date}.csv")
    fv   = PiecewiseConstantForwardVariance(*fit_xi0_from_surface(surf))
    S0   = float(surf.forwards.mean())
    return surf, fv, S0


def _reprice_lh(p, kernel, fv, S0, K_dict):
    params = LiftedHestonParams(H=p["H"], n=20, r_n=2.5,
                                 nu=p["nu"], rho=p["rho"], kernel=kernel)
    return lifted_heston_iv_surface(params, fv, S0, K_dict)


def _reprice_ab(p, kernel, fv, S0, K_dict, M_paths, seed):
    from src.calibration.optimizer import _build_ab_params
    params = _build_ab_params(p["H"], 20, p["eta"], p["rho"], kernel, 2.5)
    ivs, _ = abergomi_iv_surface(params, fv, S0, K_dict,
                                  M_paths=M_paths, qmc=False, seed=seed)
    return ivs


def _reprice_tf(p, fv, S0, K_dict, n_steps: int = 1600):
    params = TwoFactorLiftedHestonParams(
        H1=p["H1"], n1=20, r_n1=2.5, nu1=p["nu1"], rho1=p["rho1"],
        lam2=p["lam2"], theta2=p["theta2"], nu2=p["nu2"], rho2=p["rho2"],
        V2_0=p["V2_0"],
    )
    return two_factor_lh_iv_surface(params, fv, S0, K_dict, n_steps=n_steps)


def _atm_skew(ivs_dict, K_dict, F_dict, rmse_iv_max: float = 0.03):
    """ATM skew term structure |d sigma_impl/dk|_{k=0}| via SVI fit per slice.

    Slices where the SVI unweighted rmse_iv > rmse_iv_max are excluded (NaN):
    those fits failed to converge to a sensible smile and the ATM derivative
    would be unreliable. The same threshold applies to market and all models.
    """
    from src.data.svi import fit_svi_from_iv_dict
    fits = fit_svi_from_iv_dict(ivs_dict, K_dict, F_dict)
    Ts, Sk = [], []
    for t in sorted(fits):
        if t < T_MIN:
            continue
        fit = fits[t]
        if fit.rmse_iv > rmse_iv_max:
            continue
        Ts.append(t)
        Sk.append(abs(fit.atm_skew_derivative()))
    return np.array(Ts), np.array(Sk)


def _pick_mats(all_T, n=6):
    st = np.array(sorted(t for t in all_T if t >= T_MIN))
    if not len(st):
        return st
    targets = np.exp(np.linspace(np.log(st[0]), np.log(st[-1]), n))
    idx = [int(np.argmin(np.abs(st - t))) for t in targets]
    return st[sorted(set(idx))]


def _iv_panel(title, mkt_ivs, K_dict, F_dict, model_dict, S0, T_plot):
    n = len(T_plot); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=12)
    for ax, T in zip(axes[:n], T_plot):
        Ks = K_dict.get(T)
        if Ks is None:
            ax.set_visible(False); continue
        F = F_dict.get(T, S0)
        k = np.log(Ks / F)
        order = np.argsort(k)              # FIX: sort by log-moneyness
        k_s = k[order]
        assert np.all(np.diff(k_s) >= 0), "log-moneyness must be sorted before plotting"
        ax.plot(k_s, mkt_ivs[T][order], **MKT)
        for name, ivs_all in model_dict.items():
            if T not in ivs_all:
                continue
            iv_m = ivs_all[T][order]       # apply same ordering to model IVs
            fin = np.isfinite(iv_m) & (iv_m > 0.01) & (iv_m < 1.5)
            if fin.sum() > 1:
                ax.plot(k_s[fin], iv_m[fin], **STYLES.get(name, {}))
        ax.set_title(f"T={T:.3f}y", fontsize=9)
        ax.set_xlabel("log(K/F)"); ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(model_dict) + 1, 6),
               bbox_to_anchor=(0.5, -0.04), fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path.name}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    default="2024-08-05")
    parser.add_argument("--test",     default="2024-08-06")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-ab",    action="store_true")
    parser.add_argument("--M-paths",  type=int, default=100_000)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--out",      type=Path,
                        default=_ROOT / "results/07b_combined_plots")
    args = parser.parse_args()

    out_dir = args.out / f"{args.train}_{args.test}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load surfaces ─────────────────────────────────────────────────────────
    print("Loading surfaces ...")
    surf_tr, fv_tr, S0_tr = _load_surface(args.train, args.data_dir)
    surf_te, fv_te, S0_te = _load_surface(args.test,  args.data_dir)
    mkt_tr = surf_tr.ivs_per_T(); K_tr = surf_tr.strikes_per_T(); F_tr = surf_tr.forward_per_T()
    mkt_te = surf_te.ivs_per_T(); K_te = surf_te.strikes_per_T(); F_te = surf_te.forward_per_T()

    sf_base = _ROOT / "results/07_temporal_oos"
    tf_base = _ROOT / "results/07b_temporal_oos_two_factor"

    KERNEL_MAP = {
        "LH-geo": ("lh", "geometric"),
        "LH-L2":  ("lh", "l2"),
        "aB-L2":  ("ab", "l2"),
        "aB-geo": ("ab", "geometric"),
    }

    # ── Loop over lambda values ───────────────────────────────────────────────
    bar_data = {}   # {label: {iv_IS, sk_IS, iv_OOS, sk_OOS}}

    for lam_tag in ["lam0.0", "laminf"]:
        lam_label = "λ=0" if lam_tag == "lam0.0" else "λ=∞"
        print(f"\n{'='*50}")
        print(f"Processing {lam_label} ...")

        # ── Load single-factor results ────────────────────────────────────────
        sf_dir = sf_base / f"{args.train}_{args.test}_{args.seed}_{lam_tag}"
        if not sf_dir.exists():
            print(f"  Single-factor results not found: {sf_dir}")
            print(f"  Run: python scripts/07_temporal_oos.py --train {args.train} "
                  f"--test {args.test} {'--lam inf' if lam_tag=='laminf' else ''}")
            continue

        with open(sf_dir / "temporal_oos_results.json") as f:
            sf_data = json.load(f)
        sf_results = sf_data["results"]

        # ── Load two-factor results ───────────────────────────────────────────
        tf_dir = tf_base / f"{args.train}_{args.test}_{args.seed}_{lam_tag}"
        tf_params = None
        if tf_dir.exists():
            with open(tf_dir / "temporal_oos_results.json") as f:
                tf_data = json.load(f)
            tf_params = tf_data["results"]["TF-LH"]["params"]
            tf_is  = tf_data["results"]["TF-LH"]["in_sample"]
            tf_oos = tf_data["results"]["TF-LH"]["temporal_oos"]
        else:
            print(f"  Two-factor results not found: {tf_dir}")
            print(f"  Run: python scripts/07b_temporal_oos_two_factor.py "
                  f"--train {args.train} --test {args.test} "
                  f"{'--lam inf' if lam_tag=='laminf' else ''}")

        # ── Reprice all models ────────────────────────────────────────────────
        model_tr, model_te = {}, {}

        for name, rec in sf_results.items():
            p = rec["params"]
            family, kernel = KERNEL_MAP.get(name, (None, None))
            if family is None:
                continue
            if args.no_ab and family == "ab":
                print(f"  Skipping {name} (--no-ab)")
                continue
            print(f"  Repricing {name} ...")
            if family == "lh":
                model_tr[name] = _reprice_lh(p, kernel, fv_tr, S0_tr, K_tr)
                model_te[name] = _reprice_lh(p, kernel, fv_te, S0_te, K_te)
            else:
                model_tr[name] = _reprice_ab(p, kernel, fv_tr, S0_tr, K_tr,
                                              args.M_paths, args.seed + 10)
                model_te[name] = _reprice_ab(p, kernel, fv_te, S0_te, K_te,
                                              args.M_paths, args.seed + 20)
            # store bar data
            bar_data[f"{name}\n{lam_label}"] = {
                "iv_IS":  rec["in_sample"]["iv_rmse"],
                "sk_IS":  rec["in_sample"]["skew_rmse"],
                "iv_OOS": rec["temporal_oos"]["iv_rmse"],
                "sk_OOS": rec["temporal_oos"]["skew_rmse"],
            }

        if tf_params is not None:
            print("  Repricing TF-LH ...")
            model_tr["TF-LH"] = _reprice_tf(tf_params, fv_tr, S0_tr, K_tr)
            model_te["TF-LH"] = _reprice_tf(tf_params, fv_te, S0_te, K_te)
            bar_data[f"TF-LH\n{lam_label}"] = {
                "iv_IS":  tf_is["iv_rmse"],  "sk_IS":  tf_is["skew_rmse"],
                "iv_OOS": tf_oos["iv_rmse"], "sk_OOS": tf_oos["skew_rmse"],
            }

        T_tr = _pick_mats(list(mkt_tr.keys()))
        T_te = _pick_mats(list(mkt_te.keys()))

        # ── Figure A: IV slices train (all models) ────────────────────────────
        fig = _iv_panel(
            f"IV slices — {args.train} (IS)  [{lam_label}]",
            mkt_tr, K_tr, F_tr, model_tr, S0_tr, T_tr)
        _save(fig, out_dir / f"iv_slices_train_{lam_tag}.png")

        # ── Figure B: IV slices test (all models) ─────────────────────────────
        fig = _iv_panel(
            f"IV slices — {args.test} (temporal OOS)  [{lam_label}]",
            mkt_te, K_te, F_te, model_te, S0_te, T_te)
        _save(fig, out_dir / f"iv_slices_test_{lam_tag}.png")

        # ── Figure C: ATM skew (all models, both dates) ───────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(f"ATM skew $|\\partial\\sigma/\\partial k|_{{k=0}}|$  [{lam_label}]", fontsize=12)
        for ax, date, mkt_ivs, K_d, F_d, m_ivs in [
            (axes[0], args.train, mkt_tr, K_tr, F_tr, model_tr),
            (axes[1], args.test,  mkt_te, K_te, F_te, model_te),
        ]:
            Tm, Sm = _atm_skew(mkt_ivs, K_d, F_d)
            ax.plot(Tm, Sm, **MKT)
            for name, ivs_d in m_ivs.items():
                T_m, S_m = _atm_skew(ivs_d, K_d, F_d)
                ax.plot(T_m, S_m, **STYLES.get(name, {}))
            ax.set_xscale("log")
            ax.set_xlabel("T (years)")
            ax.set_ylabel(r"$|\partial \sigma_{\rm impl}/\partial k|_{k=0}|$")
            lbl = "IS" if date == args.train else "Temporal OOS"
            ax.set_title(f"{date} ({lbl})")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, out_dir / f"atm_skew_{lam_tag}.png")

        # ── Figure D: TF-LH only (IV slices, both dates) ─────────────────────
        if "TF-LH" in model_tr:
            tf_only_tr = {"TF-LH": model_tr["TF-LH"]}
            tf_only_te = {"TF-LH": model_te["TF-LH"]}
            n_cols = max(len(T_tr), len(T_te))
            fig, axes = plt.subplots(2, n_cols,
                                      figsize=(4 * n_cols, 7),
                                      constrained_layout=True)
            fig.suptitle(f"Two-factor LH  [{lam_label}]", fontsize=12)
            for row, (mkt, K_d, F_d, S0, label, T_row, current_dict) in enumerate([
                (mkt_tr, K_tr, F_tr, S0_tr, f"IS ({args.train})",   T_tr, tf_only_tr),
                (mkt_te, K_te, F_te, S0_te, f"t-OOS ({args.test})", T_te, tf_only_te),
            ]):
                for col, T in enumerate(T_row):
                    ax = axes[row][col]
                    Ks = K_d.get(T)
                    if Ks is None:
                        ax.set_visible(False); continue
                    F = F_d.get(T, S0)
                    k = np.log(Ks / F)
                    order = np.argsort(k)
                    k_s = k[order]
                    assert np.all(np.diff(k_s) >= 0), "log-moneyness must be sorted before plotting"
                    ax.plot(k_s, mkt[T][order], **MKT)
                    iv_m = current_dict["TF-LH"].get(T)
                    if iv_m is not None:
                        iv_m = iv_m[order]
                        fin = np.isfinite(iv_m) & (iv_m > 0.01) & (iv_m < 1.5)
                        if fin.sum() > 1:
                            ax.plot(k_s[fin], iv_m[fin], **STYLES["TF-LH"])
                    ax.set_title(f"T={T:.3f}y", fontsize=8)
                    ax.set_xlabel("log(K/F)", fontsize=7)
                    if col == 0:
                        ax.set_ylabel(f"IV\n{label}", fontsize=7)
                    ax.grid(True, alpha=0.3)
                for col in range(len(T_row), n_cols):
                    axes[row][col].set_visible(False)
            _save(fig, out_dir / f"tf_lh_only_{lam_tag}.png")

    # ── Figure E: Summary bar chart (all models, both lambdas) ────────────────
    if bar_data:
        labels  = list(bar_data.keys())
        iv_is   = [bar_data[l]["iv_IS"]  for l in labels]
        sk_is   = [bar_data[l]["sk_IS"]  for l in labels]
        iv_oos  = [bar_data[l]["iv_OOS"] for l in labels]
        sk_oos  = [bar_data[l]["sk_OOS"] for l in labels]

        x = np.arange(len(labels)); w = 0.18
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(labels)*1.8), 5),
                                        constrained_layout=True)
        fig.suptitle(f"IS vs Temporal OOS — all models  "
                     f"({args.train} → {args.test})", fontsize=12)

        c_is  = "steelblue"; c_oos = "lightsteelblue"
        for ax, vals_is, vals_oos, ylabel in [
            (ax1, iv_is,  iv_oos,  "IV RMSE"),
            (ax2, sk_is,  sk_oos,  "Skew RMSE"),
        ]:
            for i in range(len(labels)):
                ax.bar(x[i]-w/2, vals_is[i],  width=w, color=c_is,
                       label="IS"  if i==0 else "_")
                ax.bar(x[i]+w/2, vals_oos[i], width=w, color=c_oos,
                       label="OOS" if i==0 else "_", edgecolor="grey", lw=0.5)
                ax.text(x[i]-w/2, vals_is[i]  + 0.001,
                        f"{vals_is[i]:.3f}",  ha="center", va="bottom", fontsize=6)
                ax.text(x[i]+w/2, vals_oos[i] + 0.001,
                        f"{vals_oos[i]:.3f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel(ylabel); ax.set_title(ylabel)
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

        _save(fig, out_dir / "bars_all_models.png")

    print(f"\nAll figures saved to {out_dir}")
    print("  iv_slices_train_lam0.0.png / _laminf.png")
    print("  iv_slices_test_lam0.0.png  / _laminf.png")
    print("  atm_skew_lam0.0.png        / _laminf.png")
    print("  tf_lh_only_lam0.0.png      / _laminf.png")
    print("  bars_all_models.png        — all models × both lambdas")


if __name__ == "__main__":
    main()
