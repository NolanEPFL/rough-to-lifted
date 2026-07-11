"""scripts/13_symmetric_skew_rmse_plots.py — 07b-style ATM-skew + RMSE-bar figures
for the symmetric-split run.

Reproduces the two headline plots of scripts/plot_07b_combined.py — the ATM-skew
term structure |∂σ/∂k|_{k=0}| (IS + temporal-OOS panels, SVI per slice with the
rmse_iv>0.03 exclusion and T≥1/52 filter) and the IS-vs-OOS IV-RMSE / Skew-RMSE bar
chart — but with the NEW symmetric-split model (SYM-TF) added alongside the four
single-factor baselines and the committed (clipped) two-factor TF-LH, for all three
train/test date pairs × {λ=0, λ=∞}.

Baselines/TF-LH are repriced UNCHANGED from their committed params (results/07,
results/07b); SYM-TF from the calibrated fits in results/09_symmetric. Bar metrics are
read verbatim from the committed JSONs (no recomputation). NEW WORK; nothing modified.

Usage:
    python scripts/13_symmetric_skew_rmse_plots.py            # all 3 date pairs
    python scripts/13_symmetric_skew_rmse_plots.py --no-ab    # skip slow aB MC repricing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _symmetric_common as C
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.pricing import abergomi_iv_surface
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import two_factor_lh_iv_surface
from src.two_factor_symmetric.pricing import symmetric_iv_surface
from src.data.svi import fit_svi_from_iv_dict

OUTDIR = _ROOT / "results" / "09_symmetric" / "skew_rmse_plots"
T_MIN = 1.0 / 52

# 07b styles, plus the committed clipped TF-LH and the NEW SYM-TF.
STYLES = {
    "LH-geo": dict(color="blue",   ls="-",  lw=1.5, label="LH-geo"),
    "LH-L2":  dict(color="blue",   ls="--", lw=1.2, label="LH-L²"),
    "aB-L2":  dict(color="red",    ls="--", lw=1.5, label="aB-L²"),
    "aB-geo": dict(color="green",  ls=":",  lw=1.5, label="aB-geo"),
    "TF-LH":  dict(color="purple", ls="--", lw=1.5, label="TF-LH (clipped)"),
    "SYM-TF": dict(color="darkorange", ls="-", lw=2.6, label="SYM-TF (new)"),
}
MKT = dict(color="black", marker="o", ms=3.5, lw=0, zorder=5, label="Market")
BAR_ORDER = ["LH-geo", "LH-L2", "aB-L2", "aB-geo", "TF-LH", "SYM-TF"]


def _reprice_lh(p, kernel, fv, S0, K):
    return lifted_heston_iv_surface(
        LiftedHestonParams(H=p["H"], n=20, r_n=2.5, nu=p["nu"], rho=p["rho"], kernel=kernel),
        fv, S0, K)


def _reprice_ab(p, kernel, fv, S0, K, M_paths, seed):
    from src.calibration.optimizer import _build_ab_params
    ivs, _ = abergomi_iv_surface(_build_ab_params(p["H"], 20, p["eta"], p["rho"], kernel, 2.5),
                                 fv, S0, K, M_paths=M_paths, qmc=False, seed=seed)
    return ivs


def _reprice_tf(p, fv, S0, K, n_steps=1600):
    return two_factor_lh_iv_surface(
        TwoFactorLiftedHestonParams(H1=p["H1"], n1=20, r_n1=2.5, nu1=p["nu1"], rho1=p["rho1"],
                                    lam2=p["lam2"], theta2=p["theta2"], nu2=p["nu2"],
                                    rho2=p["rho2"], V2_0=p["V2_0"]),
        fv, S0, K, n_steps=n_steps)


def _reprice_sym(p, fv, S0, K, n_steps=1600):
    return symmetric_iv_surface(
        C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"]),
        fv, S0, K, n_steps=n_steps)


def _atm_skew(ivs_dict, K_dict, F_dict, rmse_iv_max=0.03):
    """|dσ/dk|_{k=0} via SVI per slice; drop slices with SVI rmse_iv>0.03 or T<1/52."""
    fits = fit_svi_from_iv_dict(ivs_dict, K_dict, F_dict)
    Ts, Sk = [], []
    for t in sorted(fits):
        if t < T_MIN or fits[t].rmse_iv > rmse_iv_max:
            continue
        Ts.append(t); Sk.append(abs(fits[t].atm_skew_derivative()))
    return np.array(Ts), np.array(Sk)


def _load_sf(train, test, lam_tag):
    f = C.RES07 / f"{train}_{test}_42_lam{lam_tag}/temporal_oos_results.json"
    return json.load(open(f))["results"] if f.exists() else {}


def _load_tf(train, test, lam_tag):
    f = C.RES07B / f"{train}_{test}_42_lam{lam_tag}/temporal_oos_results.json"
    return json.load(open(f))["results"]["TF-LH"] if f.exists() else None


def _load_sym(train, lt):
    f = _ROOT / "results/09_symmetric" / f"sym_{train}_lam{lt}_s42.json"
    return json.load(open(f)) if f.exists() else None


def process(train, test, no_ab, M_paths):
    surf_tr, fv_tr, S0_tr = C.load_surface(train)
    surf_te, fv_te, S0_te = C.load_surface(test)
    mkt_tr, K_tr, F_tr = surf_tr.ivs_per_T(), surf_tr.strikes_per_T(), surf_tr.forward_per_T()
    mkt_te, K_te, F_te = surf_te.ivs_per_T(), surf_te.strikes_per_T(), surf_te.forward_per_T()
    out_dir = OUTDIR / f"{train}_{test}"
    out_dir.mkdir(parents=True, exist_ok=True)
    KERNEL_MAP = {"LH-geo": ("lh", "geometric"), "LH-L2": ("lh", "l2"),
                  "aB-L2": ("ab", "l2"), "aB-geo": ("ab", "geometric")}

    for lt, lam_tag, lam_label in [("0", "0.0", "λ=0"), ("inf", "inf", "λ=∞")]:
        print(f"\n[{train}→{test}] {lam_label}")
        sf = _load_sf(train, test, lam_tag)
        tf = _load_tf(train, test, lam_tag)
        sym = _load_sym(train, lt)
        model_tr, model_te, bars = {}, {}, {}

        for name, rec in sf.items():
            fam, kern = KERNEL_MAP.get(name, (None, None))
            if fam is None or (no_ab and fam == "ab"):
                continue
            print(f"  reprice {name}")
            if fam == "lh":
                model_tr[name] = _reprice_lh(rec["params"], kern, fv_tr, S0_tr, K_tr)
                model_te[name] = _reprice_lh(rec["params"], kern, fv_te, S0_te, K_te)
            else:
                model_tr[name] = _reprice_ab(rec["params"], kern, fv_tr, S0_tr, K_tr, M_paths, 52)
                model_te[name] = _reprice_ab(rec["params"], kern, fv_te, S0_te, K_te, M_paths, 62)
            bars[name] = dict(iv_IS=rec["in_sample"]["iv_rmse"], sk_IS=rec["in_sample"]["skew_rmse"],
                              iv_OOS=rec["temporal_oos"]["iv_rmse"], sk_OOS=rec["temporal_oos"]["skew_rmse"])
        if tf is not None:
            print("  reprice TF-LH (committed clipped)")
            model_tr["TF-LH"] = _reprice_tf(tf["params"], fv_tr, S0_tr, K_tr)
            model_te["TF-LH"] = _reprice_tf(tf["params"], fv_te, S0_te, K_te)
            bars["TF-LH"] = dict(iv_IS=tf["in_sample"]["iv_rmse"], sk_IS=tf["in_sample"]["skew_rmse"],
                                 iv_OOS=tf["temporal_oos"]["iv_rmse"], sk_OOS=tf["temporal_oos"]["skew_rmse"])
        if sym is not None:
            print("  reprice SYM-TF (new)")
            model_tr["SYM-TF"] = _reprice_sym(sym["params"], fv_tr, S0_tr, K_tr)
            model_te["SYM-TF"] = _reprice_sym(sym["params"], fv_te, S0_te, K_te)
            bars["SYM-TF"] = dict(iv_IS=sym["in_sample"]["iv_rmse"], sk_IS=sym["in_sample"]["skew_rmse"],
                                  iv_OOS=sym["temporal_oos"]["iv_rmse"], sk_OOS=sym["temporal_oos"]["skew_rmse"])

        # ── ATM-skew figure (07b Figure C style) ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(rf"ATM skew $|\partial\sigma/\partial k|_{{k=0}}|$  [{lam_label}]   "
                     f"{train} → {test}", fontsize=12)
        for ax, date, mk, Kd, Fd, md, lbl in [
            (axes[0], train, mkt_tr, K_tr, F_tr, model_tr, "IS"),
            (axes[1], test, mkt_te, K_te, F_te, model_te, "Temporal OOS")]:
            Tm, Sm = _atm_skew(mk, Kd, Fd)
            ax.plot(Tm, Sm, **MKT)
            for name in BAR_ORDER:
                if name in md:
                    Tx, Sx = _atm_skew(md[name], Kd, Fd)
                    ax.plot(Tx, Sx, **STYLES[name])
            ax.set_xscale("log"); ax.set_xlabel("T (years)")
            ax.set_ylabel(r"$|\partial\sigma_{\rm impl}/\partial k|_{k=0}|$")
            ax.set_title(f"{date} ({lbl})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"atm_skew_lam{lam_tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved atm_skew_lam{lam_tag}.png")

        # ── RMSE bars (07b Figure E style) ─────────────────────────────────────
        labels = [n for n in BAR_ORDER if n in bars]
        x = np.arange(len(labels)); w = 0.38
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(11, len(labels) * 1.7), 4.8),
                                     constrained_layout=True)
        fig.suptitle(f"IS vs Temporal-OOS RMSE — {train} → {test}  [{lam_label}]", fontsize=12)
        for ax, kis, koos, ylab in [(a1, "iv_IS", "iv_OOS", "IV RMSE"),
                                    (a2, "sk_IS", "sk_OOS", "Skew RMSE")]:
            vis = [bars[l][kis] for l in labels]; voos = [bars[l][koos] for l in labels]
            ax.bar(x - w / 2, vis, width=w, color="steelblue", label="IS")
            ax.bar(x + w / 2, voos, width=w, color="lightsteelblue", edgecolor="grey", lw=0.5, label="OOS")
            for i in range(len(labels)):
                ax.text(x[i] - w / 2, vis[i], f"{vis[i]:.3f}", ha="center", va="bottom", fontsize=6)
                ax.text(x[i] + w / 2, voos[i], f"{voos[i]:.3f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
            ax.set_ylabel(ylab); ax.set_title(ylab); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.savefig(out_dir / f"rmse_bars_lam{lam_tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved rmse_bars_lam{lam_tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ab", action="store_true")
    ap.add_argument("--M-paths", type=int, default=100_000)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for train, test in C.DATE_PAIRS.items():
        process(train, test, a.no_ab, a.M_paths)
    print(f"\nAll figures under {OUTDIR}")


if __name__ == "__main__":
    main()
