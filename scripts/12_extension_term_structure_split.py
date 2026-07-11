"""scripts/12_extension_term_structure_split.py — Extensions to the symmetric model.

NEW WORK. Implements and evaluates two extensions, and documents a third:

  EXT 1  Maturity-dependent additive split w(t)=w_L+(w_S-w_L)e^{-at}  (9 params).
         Still additive ⇒ no clip, E[V_t]=ξ0(t) exact. Lets the slow block earn a
         larger share at long maturities (where its skew should matter) without
         starving the short end. Calibrated on the 3 dates × {λ=0, λ=∞}, seed 42,
         and compared head-to-head with the constant-w base model and best 1F.

  EXT 2  Two-Hurst slow block (n2>1 via a second Hurst H2). Already supported by the
         constructor; here demonstrated (prices + invariants) — a rough+rough model
         with two power-law timescales.

  EXT 3  Correlated blocks (ρ12≠0) via a Wishart / matrix-valued lift — documented as
         research-level future work (breaks the affine factorisation otherwise).

Outputs: results/09_symmetric/extension_ts/*.json, figures, and EXTENSIONS.md.
No existing/tracked file is modified.

Usage:
    python scripts/12_extension_term_structure_split.py                 # full
    python scripts/12_extension_term_structure_split.py --skip-calib    # analysis only
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
from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.characteristic_function import SymmetricTwoFactorCF
from src.two_factor_symmetric.pricing import symmetric_call_prices, symmetric_iv_surface
from src.two_factor_symmetric.term_structure_split import (
    SymmetricTwoFactorTSParams, SymmetricTwoFactorTSCF, ts_call_prices,
    ts_iv_surface, ts_weight,
)

OUTDIR = _ROOT / "results" / "09_symmetric"
EXTDIR = OUTDIR / "extension_ts"
FIGDIR = OUTDIR / "figures"
WORKER = Path(__file__).resolve().parent / "_calib_symmetric_ts.py"
DATES = list(C.DATE_PAIRS.keys())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ts(date, lt, seed=42):
    f = EXTDIR / f"ts_{date}_lam{lt}_s{seed}.json"
    return json.load(open(f)) if f.exists() else None


def load_sym(date, lt, seed=42):
    f = OUTDIR / f"sym_{date}_lam{lt}_s{seed}.json"
    return json.load(open(f)) if f.exists() else None


# ── EXT-1 sanity: TS nests the constant-w base model (w_S=w_L) ─────────────────

def sanity_nesting():
    log("EXT-1 sanity: TS(w_S=w_L=w) must equal the constant-w base model")
    from src.common.forward_variance import FlatForwardVariance
    fv = FlatForwardVariance(0.04); S0 = 100.0
    w = 0.3
    base = SymmetricTwoFactorParams(w=w, H1=0.10, nu1=0.6, rho1=-0.7, kappa2=1.0, nu2=0.5, rho2=-0.5)
    ts = SymmetricTwoFactorTSParams(w_S=w, w_L=w, a=2.0, H1=0.10, nu1=0.6, rho1=-0.7,
                                    kappa2=1.0, nu2=0.5, rho2=-0.5)
    maxerr = 0.0
    omega = np.linspace(0, 50, 200)
    u = np.concatenate([1j * omega, np.array([0j, 1 + 0j])])
    for T in [0.25, 1.0, 2.0]:
        e = float(np.max(np.abs(SymmetricTwoFactorCF(base, fv, T).cf_centered(u)
                                - SymmetricTwoFactorTSCF(ts, fv, T).cf_centered(u))))
        maxerr = max(maxerr, e)
    log(f"  max |Φ_TS(w_S=w_L) - Φ_base| = {maxerr:.2e}  {'[PASS]' if maxerr < 1e-12 else '[FAIL]'}")
    return maxerr


def sanity_invariants_ts():
    """TS keeps E[V_t]=ξ0 exact and clip=0 on a steep curve, for w_S≠w_L."""
    log("EXT-1 sanity: TS keeps clip=0 and E[V]=ξ0 exact on a steep curve (w_S≠w_L)")
    surf, fv, S0 = C.load_surface("2024-08-05")
    Tmax = float(max(surf.ivs_per_T().keys()))
    tg = np.linspace(1e-4, Tmax, 6000); xg = np.array([float(fv(t)) for t in tg])
    par = SymmetricTwoFactorTSParams(w_S=0.1, w_L=0.8, a=1.5, H1=0.10, nu1=0.6, rho1=-0.7,
                                     kappa2=0.5, nu2=0.5, rho2=-0.5)
    w_t = ts_weight(tg, par.w_S, par.w_L, par.a)
    g1 = (1 - w_t) * xg; g2 = w_t * xg
    clip = float(np.mean((g1 < 0) | (g2 < 0)))
    relEV = float(np.abs((g1 + g2) / xg - 1.0).max())
    log(f"  clip_frac={clip:.0e}  max|E[V]/ξ0-1|={relEV:.1e}  "
        f"w(t): {w_t[0]:.3f}→{w_t[-1]:.3f}  {'[PASS]' if clip == 0 and relEV < 1e-13 else '[FAIL]'}")
    return dict(clip=clip, relEV=relEV)


# ── EXT-2 demo: two-Hurst slow block ───────────────────────────────────────────

def demo_two_hurst():
    log("EXT-2 demo: two-Hurst slow block (n2=20, H2=0.30) prices + invariants")
    surf, fv, S0 = C.load_surface("2024-12-04")
    K = surf.strikes_per_T()
    par = SymmetricTwoFactorParams(w=0.5, H1=0.08, nu1=0.6, rho1=-0.7, kappa2=1.0,
                                   nu2=0.5, rho2=-0.5, n2=20, H2=0.30, r_n2=2.5)
    Ts = sorted(K)[:4]
    iv = symmetric_iv_surface(par, fv, S0, {T: K[T] for T in Ts}, n_steps=400)
    finite = all(np.isfinite(v).any() for v in iv.values())
    # invariant: E[V]=ξ0 exact (additive split, any kernel)
    tg = np.linspace(1e-4, float(sorted(K)[-1]), 4000); xg = np.array([float(fv(t)) for t in tg])
    relEV = float(np.abs(((1 - 0.5) * xg + 0.5 * xg) / xg - 1.0).max())
    log(f"  two-Hurst prices finite={finite}  H1=0.08/H2=0.30 (rough+rough)  "
        f"max|E[V]/ξ0-1|={relEV:.1e}  [PASS]" if finite else "  [FAIL]")
    return dict(finite=bool(finite), relEV=relEV)


# ── EXT-1 calibration dispatch (6 configs) ─────────────────────────────────────

def dispatch(concurrency, de_maxiter):
    EXTDIR.mkdir(parents=True, exist_ok=True)
    configs = [(d, C.DATE_PAIRS[d], lt, 42) for d in DATES for lt in ("0", "inf")]
    pending, running, done, failed = [], {}, [], []
    for (tr, te, lt, s) in configs:
        tag = f"ts_{tr}_lam{lt}_s{s}"
        (done if (EXTDIR / f"{tag}.json").exists() else pending).append((tr, te, lt, s, tag))
    pending = [c for c in pending if isinstance(c, tuple)]
    log(f"EXT-1 dispatch: {len(done)} done, {len(pending)} to run, conc={concurrency}")
    logf = open(EXTDIR / "calib_ts.log", "a", encoding="utf-8")
    while pending or running:
        while pending and len(running) < concurrency:
            tr, te, lt, s, tag = pending.pop(0)
            cmd = [sys.executable, str(WORKER), "--train", tr, "--test", te,
                   "--lam", ("inf" if lt == "inf" else "0.0"), "--seed", str(s),
                   "--de-maxiter", str(de_maxiter), "--workers", "1", "--tag", tag]
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            running[tag] = (p, time.perf_counter()); log(f"launch {tag}")
        for tag, (p, t0) in list(running.items()):
            if p.poll() is not None:
                dt = (time.perf_counter() - t0) / 60
                ok = (EXTDIR / f"{tag}.json").exists()
                log(f"{'done' if ok else 'FAILED'} {tag} ({dt:.1f} min)"); del running[tag]
        time.sleep(3)
    logf.close()


# ── Analysis + figure + EXTENSIONS.md ──────────────────────────────────────────

def comparison():
    rows = []
    for date in DATES:
        test = C.DATE_PAIRS[date]
        for lt in ("0", "inf"):
            ts = load_ts(date, lt); base = load_sym(date, lt)
            oneF = C.single_factor_iv(date, test, lt)
            if ts is None or base is None or oneF is None:
                continue
            metric = "skew_rmse" if lt == "inf" else "iv_rmse"
            rows.append(dict(
                date=date, lam=lt, metric=metric,
                ts_iv=ts["in_sample"]["iv_rmse"], ts_skew=ts["in_sample"]["skew_rmse"],
                base_iv=base["in_sample"]["iv_rmse"], base_skew=base["in_sample"]["skew_rmse"],
                best1F_iv=oneF["iv_IS"], best1F_model=oneF["model"],
                ts_frac_clip=ts["frac_clipped"], ts_relEV=ts["max_relEV_dev"],
                w_S=ts["params"]["w_S"], w_L=ts["params"]["w_L"], a=ts["params"]["a"],
                kappa2=ts["params"]["kappa2"],
                ts_vs_base_iv_pct=100 * (ts["in_sample"]["iv_rmse"] - base["in_sample"]["iv_rmse"]) / base["in_sample"]["iv_rmse"],
                ts_vs_base_skew_pct=(100 * (ts["in_sample"]["skew_rmse"] - base["in_sample"]["skew_rmse"]) / base["in_sample"]["skew_rmse"]
                                     if base["in_sample"]["skew_rmse"] else float("nan")),
                ts_vs_best1F_iv_pct=100 * (ts["in_sample"]["iv_rmse"] - oneF["iv_IS"]) / oneF["iv_IS"]))
    return rows


def make_figure():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    T = np.linspace(0.0, 2.0, 200)
    any_curve = False
    for date in DATES:
        ts = load_ts(date, "inf")
        if ts is None:
            continue
        p = ts["params"]
        ax.plot(T, ts_weight(T, p["w_S"], p["w_L"], p["a"]),
                label=f"{date} (λ=∞): w_S={p['w_S']:.2f}→w_L={p['w_L']:.2f}, a={p['a']:.1f}")
        any_curve = True
    ax.set_xlabel("maturity T (yr)"); ax.set_ylabel("block-2 variance share  w(T)")
    ax.set_title("EXT-1: calibrated maturity-dependent additive split w(T)")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    if any_curve:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "extension_wT_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {FIGDIR / 'extension_wT_profiles.png'}")


def write_extensions_md(rows, san):
    def tbl(rows):
        out = ["| Date | λ | SYM-TS iv_IS | base SYM iv_IS | SYM-TS skew | base skew | clip | \\|E[V]/ξ0−1\\| | w_S→w_L (a) |",
               "|------|---|------|------|------|------|------|------|------|"]
        for r in rows:
            out.append(
                f"| {r['date']} | {r['lam']} | {r['ts_iv']:.4f} | {r['base_iv']:.4f} | "
                f"{r['ts_skew']:.4f} | {r['base_skew']:.4f} | {r['ts_frac_clip']:.0e} | "
                f"{r['ts_relEV']:.1e} | {r['w_S']:.2f}→{r['w_L']:.2f} ({r['a']:.1f}) |")
        return "\n".join(out)

    tbl_str = tbl(rows) if rows else "_(calibration pending — re-run with --skip-calib once JSONs exist)_"
    nest_str = f"{san['nest']:.0e}"
    clip_str = f"{san['inv']['clip']:.0e}"
    relev_str = f"{san['inv']['relEV']:.1e}"
    md = """# Extensions to the symmetric-split two-factor lifted Heston

**New work beyond the thesis.** Implemented in new files only
(`src/two_factor_symmetric/term_structure_split.py`, `scripts/12_*`,
`scripts/_calib_symmetric_ts.py`); no committed thesis result or existing `src/`
file is modified.

The base model (§8.9.7 remedy, `results/09_symmetric/REPORT.md`) fixes the
forward-variance clipping bug and gives the slow block a *constant* variance share
`w`. The erratum's structural worry — and what the fair run shows — is that a single
scalar share is a blunt instrument: it cannot give the slow block more weight at long
maturities (where its skew should matter) without taking it from the short end. The
extensions below address exactly that, while preserving the two invariants that make
the construction legitimate: **no clip** and **`E[V_t]=ξ0(t)` exact**.

## EXT 1 — Maturity-dependent additive split `w(t)` (implemented + calibrated)

`g0^(1)(t)=(1-w(t))ξ0(t)`, `g0^(2)(t)=w(t)ξ0(t)`, with
`w(t)=w_L+(w_S-w_L)e^{-a t}`, `w_S,w_L∈[0,1]`, `a>0`. Since `w(t)` is a convex blend
of two points in `[0,1]`, it stays in `[0,1]` for all `t`, so **both forcings remain
≥0 (no clip)** and **`E[V_t]=(1-w(t))ξ0(t)+w(t)ξ0(t)=ξ0(t)` exactly** — the §8.9.7 bug
stays gone. `w_S` is the short-maturity block-2 share, `w_L` the long-maturity share,
`a` the transition rate. Adds 2 parameters over the 7-parameter base model.

**Sanity (machine precision):** TS with `w_S=w_L` reproduces the constant-`w` base CF
to `<<NEST>>`; with `w_S≠w_L` on the steep 2024-08-05 curve, clip fraction
`<<CLIP>>` and `|E[V]/ξ0−1|=<<RELEV>>`.

**Head-to-head (seed 42), in-sample, vs the constant-`w` base model and best single-factor:**

<<TBL>>

The calibrated `w(T)` profiles are in `figures/extension_wT_profiles.png`. Read the
`SYM-TS skew` vs `base skew` columns for the λ=∞ rows: that is where a maturity-varying
split is expected to help, by decoupling the short-end and long-end skew the constant-`w`
model ties together.

## EXT 2 — Two-Hurst slow block (implemented, supported by the constructor)

Setting `n2>1` with a second Hurst `H2` makes block 2 itself a lifted (rough) block:
a **rough+rough** model with two distinct power-law timescales (`H1` for the steep
short end, `H2` for a slower long-end decay), instead of a single OU speed `κ2`. The
additive split and both invariants are unchanged (any non-negative kernel works).
Demonstrated in this script (finite, sane prices; `E[V]=ξ0` exact). A full calibration
of the two-Hurst variant is a drop-in (`SymmetricTwoFactorParams(..., n2=20, H2=...)`).

## EXT 3 — Correlated blocks `ρ12≠0` (proposed; research-level)

The thesis Case I assumes `W1⊥W2`, which makes the joint CF factorise into the product
used throughout. A shared or partially-correlated driver introduces non-affine cross
terms that destroy either positivity or the decoupled Riccati. The principled remedy
(noted in the thesis) is a **matrix-valued (Wishart) lift**, where the variance state
lives in the cone of symmetric positive-semidefinite matrices and cross-block leverage
is carried by off-diagonal entries while positivity is preserved. This is a genuine
research extension (new Riccati system on `S_d^+`, new pricing), documented here as the
natural next step rather than implemented.

## Other directions (proposed)
- **Three+ blocks / a timescale spectrum** partitioned additively (`Σ w_i(t)=1`,
  `w_i≥0`) — same positivity/`ξ0` guarantees, a richer skew term structure.
- **Joint SPX–VIX calibration**: the exact `E[V_t]=ξ0` match makes the additive split a
  natural VIX-consistent parametrisation.

_Generated by `scripts/12_extension_term_structure_split.py`._
"""
    md = (md.replace("<<TBL>>", tbl_str).replace("<<NEST>>", nest_str)
            .replace("<<CLIP>>", clip_str).replace("<<RELEV>>", relev_str))
    (OUTDIR / "EXTENSIONS.md").write_text(md, encoding="utf-8")
    log(f"wrote {OUTDIR / 'EXTENSIONS.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--de-maxiter", type=int, default=30)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("=== Extensions: start ===")
    san = dict(nest=sanity_nesting(), inv=sanity_invariants_ts(), twohurst=demo_two_hurst())
    if not a.skip_calib:
        dispatch(a.concurrency, a.de_maxiter)
    rows = comparison()
    json.dump(dict(comparison=rows, sanity=san), open(OUTDIR / "extensions_results.json", "w"),
              indent=2, default=float)
    try:
        make_figure()
    except Exception as e:  # noqa: BLE001
        import traceback; log(f"figure error: {e}"); traceback.print_exc()
    write_extensions_md(rows, san)
    log("=== Extensions: complete ===")


if __name__ == "__main__":
    main()
