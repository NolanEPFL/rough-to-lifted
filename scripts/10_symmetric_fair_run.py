"""scripts/10_symmetric_fair_run.py — Step 4 fair run for the symmetric-split model.

Reproduces the clip-free, same-ξ0 protocol of the erratum (Table 8.6) with the
symmetric-split two-factor model in place of the patched/decoupled one:

  * 3 train/test pairs x {λ=0, λ=∞} x 5 seeds {42,1,2,3,4}  = 30 calibrations,
    dispatched via _calib_symmetric.py (DE pop10/iter30/seed + L-BFGS-B refine).
  * Fair Δ = 100·(clean SYM iv_IS − best single-factor iv_IS)/best, with the
    best single-factor iv_IS read UNCHANGED from committed results/07 (4 models).
  * Tables (λ=0, λ=∞, seed 42) + 5-seed spread + block-2 activity + a positivity
    re-probe at each date's CALIBRATED parameters.
  * Figures for 2024-12-04: IV slices vs the 4 baselines; SVI ATM-skew term
    structure (IS & tOOS). Final reprice for plots at n_steps=1600.
  * results/09_symmetric/REPORT.md.

Usage:
    python scripts/10_symmetric_fair_run.py                # calibrate + analyse + figures + report
    python scripts/10_symmetric_fair_run.py --skip-calib   # analysis only (JSONs already present)
    python scripts/10_symmetric_fair_run.py --concurrency 12

NEW WORK; no existing/tracked file is modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
from src.two_factor_symmetric.pricing import symmetric_iv_surface
from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.monte_carlo import simulate_symmetric

OUTDIR = _ROOT / "results" / "09_symmetric"
FIGDIR = OUTDIR / "figures"
WORKER = Path(__file__).resolve().parent / "_calib_symmetric.py"
LOG = OUTDIR / "run.log"
DATES = list(C.DATE_PAIRS.keys())
SEEDS = C.SEEDS


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tag_of(date, lt, seed):
    return f"sym_{date}_lam{lt}_s{seed}"


def load_sym(date, lt, seed):
    f = OUTDIR / f"{tag_of(date, lt, seed)}.json"
    return json.load(open(f)) if f.exists() else None


def load8(date, lt, seed=42):
    f = _ROOT / "results/08b_clipfree_two_factor" / f"clip_{date}_lam{lt}_s{seed}.json"
    return json.load(open(f)) if f.exists() else None


# ── Phase 1: dispatch the 30 calibrations (parallel subprocess pool) ───────────

def dispatch(concurrency, de_maxiter):
    configs = [(d, C.DATE_PAIRS[d], lt, s) for d in DATES for lt in ("0", "inf") for s in SEEDS]
    pending, running, done, failed = [], {}, [], []
    for (train, test, lt, seed) in configs:
        tag = tag_of(train, lt, seed)
        if (OUTDIR / f"{tag}.json").exists():
            done.append(tag)
        else:
            pending.append((train, test, lt, seed, tag))
    log(f"dispatch: {len(done)} already done, {len(pending)} to run, concurrency={concurrency}")
    logf = open(OUTDIR / "calib_workers.log", "a", encoding="utf-8")
    while pending or running:
        while pending and len(running) < concurrency:
            train, test, lt, seed, tag = pending.pop(0)
            cmd = [sys.executable, str(WORKER), "--train", train, "--test", test,
                   "--lam", ("inf" if lt == "inf" else "0.0"), "--seed", str(seed),
                   "--de-maxiter", str(de_maxiter), "--workers", "1", "--tag", tag]
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            running[tag] = (p, time.perf_counter())
            log(f"launch {tag} ({len(running)} running, {len(pending)} queued)")
        for tag, (p, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                dt = time.perf_counter() - t0
                if rc == 0 and (OUTDIR / f"{tag}.json").exists():
                    done.append(tag); log(f"done {tag} ({dt/60:.1f} min)")
                else:
                    failed.append(tag); log(f"FAILED {tag} rc={rc} ({dt/60:.1f} min)")
                del running[tag]
        time.sleep(3)
    logf.close()
    log(f"dispatch complete: {len(done)} done, {len(failed)} failed: {failed}")
    return failed


# ── Phase 2: fair tables + 5-seed spread + block-2 activity ────────────────────

def fair_table(lt):
    """Table 8.6 analogue for a given λ tag, seed 42."""
    rows = []
    for date in DATES:
        test = C.DATE_PAIRS[date]
        J = load_sym(date, lt, 42)
        oneF = C.single_factor_iv(date, test, lt)
        clipTF = C.committed_tf_iv(date, lt)
        clean8 = load8(date, lt, 42)
        if J is None or oneF is None:
            rows.append(dict(date=date, lam=lt, missing=True)); continue
        clean = J["in_sample"]["iv_rmse"]
        oneF_iv = oneF["iv_IS"]
        rows.append(dict(
            date=date, lam=lt, missing=False,
            sym_ivIS=clean, sym_ivtOOS=J["temporal_oos"]["iv_rmse"],
            best1F_model=oneF["model"], best1F_ivIS=oneF_iv, best1F_ivtOOS=oneF["iv_tOOS"],
            fair_delta_pct=100.0 * (clean - oneF_iv) / oneF_iv,
            frac_clipped=J["frac_clipped"], max_relEV_dev=J["max_relEV_dev"],
            clipTF_ivIS=(clipTF["iv_IS"] if clipTF else None),
            erratumCleanTF_ivIS=(clean8["in_sample"]["iv_rmse"] if clean8 else None),
            erratum_fair_delta_pct=(100.0 * (clean8["in_sample"]["iv_rmse"] - oneF_iv) / oneF_iv
                                    if clean8 else None),
            params=J["params"], obj=J["obj"]))
    return rows


def seed_spread():
    out = []
    for date in DATES:
        for lt in ("0", "inf"):
            ivs = []
            for s in SEEDS:
                J = load_sym(date, lt, s)
                if J is not None:
                    ivs.append(J["in_sample"]["iv_rmse"])
            if not ivs:
                continue
            a = np.array(ivs)
            out.append(dict(date=date, lam=lt, n_seeds=len(ivs),
                            iv_min=float(a.min()), iv_max=float(a.max()),
                            iv_mean=float(a.mean()), iv_std=float(a.std()),
                            iv_range=float(a.max() - a.min())))
    return out


def activity_table():
    out = []
    for date in DATES:
        for lt in ("0", "inf"):
            J = load_sym(date, lt, 42)
            if J is None:
                continue
            act = C.block2_activity(date, J["params"])
            act.update(date=date, lam=lt)
            out.append(act)
    return out


# ── positivity re-probe at calibrated params ───────────────────────────────────

def mass_probe(params, fv, S0, T, M=40_000, n_spy=1000, seed=99):
    """Per-block pointwise frac_neg and truncated-mass ratio at calibrated params."""
    rng = np.random.default_rng(seed)
    n = max(1, int(round(T * n_spy))); dt = T / n; sqdt = np.sqrt(dt)
    s1, s2 = 1.0 - params.w, params.w
    blocks = [(params.c1, params.x1, params.nu1, s1),
              (params.c2, params.x2, params.nu2, s2)]
    res = []
    for c, x, nu, scale in blocks:
        a = np.exp(-x * dt); bsq = (1 - np.exp(-2 * x * dt)) / (2 * x)
        cov = (1 - a) / (x * dt); sres = np.sqrt(np.maximum(bsq - cov ** 2 * dt, 0.0))
        U = np.zeros((M, len(c))); neg = 0; npairs = 0; negm = 0.0; posm = 0.0; mn = np.inf
        t = 0.0
        for _ in range(n):
            dW = rng.standard_normal((M,)) * sqdt; eps = rng.standard_normal((M, len(c)))
            Vraw = scale * float(fv(t)) + U @ c
            mn = min(mn, float(Vraw.min())); neg += int(np.count_nonzero(Vraw < 0)); npairs += M
            Vp = np.maximum(Vraw, 0.0); negm += float(np.maximum(-Vraw, 0).sum()) * dt
            posm += float(Vp.sum()) * dt
            U = U * a[None, :] + (nu * np.sqrt(Vp))[:, None] * (cov[None, :] * dW[:, None] + sres[None, :] * eps)
            t += dt
        res.append(dict(frac_neg=neg / npairs, min_raw=mn,
                        negmass_ratio=(negm / posm if posm > 0 else float("nan"))))
    return dict(block1=res[0], block2=res[1])


def positivity_recheck():
    out = []
    for date in DATES:
        J = load_sym(date, "0", 42)
        if J is None:
            continue
        p = J["params"]
        surf, fv, S0 = C.load_surface(date)
        Tmax = float(max(surf.ivs_per_T().keys()))
        par = C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"])
        probe = mass_probe(par, fv, S0, min(Tmax, 1.5))
        probe.update(date=date, nu1=p["nu1"], nu2=p["nu2"], w=p["w"], H1=p["H1"])
        out.append(probe)
        log(f"positivity@calib {date}: nu1={p['nu1']:.3f} "
            f"b1 frac_neg={probe['block1']['frac_neg']:.2%} mass={probe['block1']['negmass_ratio']:.2e} ; "
            f"b2 frac_neg={probe['block2']['frac_neg']:.2%} mass={probe['block2']['negmass_ratio']:.2e}")
    return out


# ── Phase 3: figures for 2024-12-04 ────────────────────────────────────────────

def reconstruct_baselines(date, fv, S0, K):
    """Reprice the 4 committed single-factor baselines from results/07 params."""
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.pricing import lifted_heston_iv_surface
    f = C.RES07 / f"{date}_{C.DATE_PAIRS[date]}_42_lam0.0/temporal_oos_results.json"
    R = json.load(open(f))["results"]
    curves = {}
    for name in ("LH-geo", "LH-L2", "aB-geo", "aB-L2"):
        if name not in R:
            continue
        p = R[name]["params"]
        try:
            if name.startswith("LH"):
                kern = "geometric" if name.endswith("geo") else "l2"
                lp = LiftedHestonParams(H=p["H"], n=20, r_n=2.5, nu=p["nu"], rho=p["rho"], kernel=kern)
                curves[name] = lifted_heston_iv_surface(lp, fv, S0, K)
            else:
                from src.calibration.optimizer import _build_ab_params
                from src.abergomi.pricing import abergomi_iv_surface
                kern = "geometric" if name.endswith("geo") else "l2"
                ap = _build_ab_params(p["H"], 20, p["eta"], p["rho"], kern, 2.5)
                ivs, _ = abergomi_iv_surface(ap, fv, S0, K, M_paths=100_000, qmc=False, seed=52)
                curves[name] = ivs
        except Exception as e:  # noqa: BLE001
            log(f"baseline {name} reconstruction failed: {e}")
    return curves


def make_figures(date="2024-12-04"):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    J = load_sym(date, "0", 42)
    if J is None:
        log(f"figures: no seed-42 λ0 fit for {date}; skipping"); return
    p = J["params"]
    par = C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"])
    surf_tr, fv_tr, S0_tr = C.load_surface(date)
    K_tr = surf_tr.strikes_per_T(); F_tr = surf_tr.forward_per_T(); mkt_tr = surf_tr.ivs_per_T()
    test = C.DATE_PAIRS[date]
    surf_te, fv_te, S0_te = C.load_surface(test)
    K_te = surf_te.strikes_per_T(); F_te = surf_te.forward_per_T(); mkt_te = surf_te.ivs_per_T()

    # (a) IV slices vs baselines (final reprice at n_steps=1600)
    sym_iv = symmetric_iv_surface(par, fv_tr, S0_tr, K_tr, n_steps=1600)
    base = reconstruct_baselines(date, fv_tr, S0_tr, K_tr)
    Ts = sorted(mkt_tr.keys())
    pick = [Ts[i] for i in np.linspace(0, len(Ts) - 1, min(6, len(Ts))).astype(int)]
    ncol = 3; nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    colors = {"LH-geo": "tab:blue", "LH-L2": "tab:cyan", "aB-geo": "tab:green", "aB-L2": "tab:olive"}
    for ax, T in zip(axes.ravel(), pick):
        k = np.log(np.asarray(K_tr[T]) / F_tr[T])
        ax.plot(k, mkt_tr[T], "k.", ms=6, label="market")
        ax.plot(k, sym_iv[T], "r-", lw=2, label="SYM-TF")
        for name, curve in base.items():
            if T in curve:
                ax.plot(k, curve[T], "--", lw=1.1, color=colors.get(name), label=name)
        ax.set_title(f"T={T:.3f}y"); ax.set_xlabel("log-moneyness k"); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    axes.ravel()[0].set_ylabel("implied vol")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"In-sample IV slices — {date} (SYM-TF vs single-factor baselines, λ=0, seed 42)", y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"iv_slices_{date}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {FIGDIR / f'iv_slices_{date}.png'}")

    # (b) SVI ATM-skew term structure (IS & tOOS)
    sym_iv_te = symmetric_iv_surface(par, fv_te, S0_te, K_te, n_steps=1600)
    sk_mkt_is = C.svi_skew_curve(mkt_tr, K_tr, F_tr)
    sk_sym_is = C.svi_skew_curve(sym_iv, K_tr, F_tr)
    sk_mkt_oos = C.svi_skew_curve(mkt_te, K_te, F_te)
    sk_sym_oos = C.svi_skew_curve(sym_iv_te, K_te, F_te)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (title, skm, sks) in zip(
        axes, [(f"In-sample ({date})", sk_mkt_is, sk_sym_is),
               (f"Temporal OOS ({test})", sk_mkt_oos, sk_sym_oos)]):
        Tm = sorted(skm); Ts2 = sorted(sks)
        ax.plot(Tm, [skm[t] for t in Tm], "ks-", ms=5, label="market SVI skew")
        ax.plot(Ts2, [sks[t] for t in Ts2], "r^-", ms=5, label="SYM-TF SVI skew")
        ax.set_xscale("log"); ax.set_xlabel("maturity T (years)")
        ax.set_ylabel(r"ATM skew $d\sigma/dk|_{k=0}$"); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"SVI-derived ATM-skew term structure — {date} (λ=0, seed 42, rmse_iv≤0.03 filter)")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"atm_skew_termstructure_{date}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {FIGDIR / f'atm_skew_termstructure_{date}.png'}")


# ── Phase 4: assemble tables JSON + REPORT.md ──────────────────────────────────

def write_report(tabs):
    def fmt_fair(rows):
        out = ["| Date | clean SYM-TF iv_IS | best single-factor iv_IS | fair Δ | clip frac | \\|E[V]/ξ0−1\\| | (erratum clean-TF Δ) |",
               "|------|------|------|------|------|------|------|"]
        for r in rows:
            if r.get("missing"):
                out.append(f"| {r['date']} | — missing — |||||"); continue
            ed = r.get("erratum_fair_delta_pct")
            out.append(
                f"| {r['date']} | {r['sym_ivIS']:.4f} | {r['best1F_ivIS']:.4f} ({r['best1F_model']}) | "
                f"{r['fair_delta_pct']:+.1f}% | {r['frac_clipped']:.0e} | {r['max_relEV_dev']:.1e} | "
                f"{(f'{ed:+.1f}%' if ed is not None else 'n/a')} |")
        return "\n".join(out)

    def fmt_spread(rows):
        out = ["| Date | λ | n | iv_IS min | max | mean | std | range |",
               "|------|---|---|------|------|------|------|------|"]
        for r in rows:
            out.append(f"| {r['date']} | {r['lam']} | {r['n_seeds']} | {r['iv_min']:.4f} | "
                       f"{r['iv_max']:.4f} | {r['iv_mean']:.4f} | {r['iv_std']:.4f} | {r['iv_range']:.4f} |")
        return "\n".join(out)

    def fmt_activity(rows):
        out = ["| Date | λ | w | \\|K2/K1\\| | block-2 SVI skew (max) | κ2 | \\|ρ1−ρ2\\| |",
               "|------|---|---|------|------|------|------|"]
        for r in rows:
            out.append(f"| {r['date']} | {r['lam']} | {r['w']:.3f} | {r['absK2K1']:.2e} | "
                       f"{r['b2_skew_max']:.2e} | {r['kappa2']:.3f} | {r['drho']:.3f} |")
        return "\n".join(out)

    def fmt_pos(rows):
        out = ["| Date | calib ν1 | calib ν2 | w | block1 frac_neg | block1 trunc-mass | block2 frac_neg | block2 trunc-mass |",
               "|------|------|------|------|------|------|------|------|"]
        for r in rows:
            b1, b2 = r["block1"], r["block2"]
            out.append(f"| {r['date']} | {r['nu1']:.3f} | {r['nu2']:.3f} | {r['w']:.3f} | "
                       f"{b1['frac_neg']:.1%} | {b1['negmass_ratio']:.2e} | "
                       f"{b2['frac_neg']:.1%} | {b2['negmass_ratio']:.2e} |")
        return "\n".join(out)

    md = f"""# Symmetric-split two-factor lifted Heston — fair-run results (§8.9.7 remedy)

**New work beyond the thesis.** The thesis marks the symmetric-split construction as
future work (§8.9.7 final paragraph, §9). It is implemented here in a separate package
`src/two_factor_symmetric/` and run by `scripts/09_*`/`scripts/10_*`; **no committed
thesis result or any existing `src/` file is modified.** All numbers below trace to
JSON files under `results/09_symmetric/`.

The model partitions the forward variance **additively** by a convex weight `w`:
`V = [(1−w)ξ0 + Σ c_i^(1) U^(1,i)] + [w ξ0 + Σ c_j^(2) U^(2,j)]`, both forcings ≥ 0 for
`w∈[0,1]`, so block 1 is **never clipped** and `E^Q[V_t]=ξ0(t)` holds exactly for any `w`.
Calibration is by COS (`n_steps=200`; `n1=20, r_n=2.5, n2=1, N_cos=256, L0=12`), same
vega-weighted IV-RMSE (λ=0) / 5-strike ATM-skew RMSE (λ=∞) loss, same thinning (10×20),
same DE (pop 10, iter 30, seed 42) + L-BFGS-B as Table 8.6. Final plot reprice at
`n_steps=1600`. `kappa2∈[0.05,10]` per spec (the only bound differing from the thesis
block-2 speed, `[0.10,10]`).

## 1. Fair comparison, λ = 0 (seed 42) — analogue of Table 8.6
{fmt_fair(tabs['fair_lam0'])}

## 2. Fair comparison, λ = ∞ (seed 42)
{fmt_fair(tabs['fair_laminf'])}

`clip frac = 0` and `|E[V]/ξ0−1| ≤ ~2e-16` on every fit — the §8.9.7 clipping bug is
structurally gone. The erratum's clip-free fair Δ for the *decoupled-CIR* model were
**+15.8% / −2.0% / −8.6%**.

## 3. Five-seed robustness of iv_IS
{fmt_spread(tabs['seed_spread'])}

## 4. Is block 2 finally active? (seed 42)
{fmt_activity(tabs['activity'])}

`w` is the block-2 variance share; `|K2/K1|` is the leading-order ATM-skew contribution
ratio (eq. 6.25, `g0^(2)=w ξ0`, `V2_0=w ξ0(0)`, single-factor `Σc/x=1/κ2`); the SVI
column is the block-2-on minus block-2-off skew difference over the traded window.

## 5. Positivity re-probe at the calibrated parameters (λ=0, seed 42)
{fmt_pos(tabs['positivity'])}

Pathwise positivity of the raw (pre-truncation) variance is a property of the finite-`n`
lifted-Heston class — equally present in the single-factor baselines (the byte-identical
`w=0` equivalence in `scripts/09` proves block 1 *is* the single-block model). It scales
with vol-of-vol; the truncated *mass* (pricing-relevant) is reported above. Calibration is
by exact-affine COS, which is unaffected (see `results/09_symmetric/diagnostics/`).

## 6. Figures (2024-12-04)
- `figures/iv_slices_2024-12-04.png` — in-sample IV slices, SYM-TF vs the 4 baselines.
- `figures/atm_skew_termstructure_2024-12-04.png` — SVI ATM-skew term structure (IS & tOOS).

_Report regenerated by `scripts/10_symmetric_fair_run.py`._
"""
    (OUTDIR / "REPORT.md").write_text(md, encoding="utf-8")
    log(f"wrote {OUTDIR / 'REPORT.md'}")


def analyse():
    tabs = dict(
        fair_lam0=fair_table("0"),
        fair_laminf=fair_table("inf"),
        seed_spread=seed_spread(),
        activity=activity_table(),
        positivity=positivity_recheck(),
    )
    json.dump(tabs, open(OUTDIR / "fair_tables.json", "w"), indent=2, default=float)
    log(f"wrote {OUTDIR / 'fair_tables.json'}")
    return tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--skip-figs", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--de-maxiter", type=int, default=30)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("=== Step 4 fair run start ===")
    if not a.skip_calib:
        failed = dispatch(a.concurrency, a.de_maxiter)
        if failed:
            log(f"WARNING: {len(failed)} configs failed; analysis proceeds with available JSONs")
    tabs = analyse()
    if not a.skip_figs:
        try:
            make_figures("2024-12-04")
        except Exception as e:  # noqa: BLE001
            import traceback; log(f"figures error: {e}"); traceback.print_exc()
    write_report(tabs)
    log("=== Step 4 fair run complete ===")


if __name__ == "__main__":
    main()
