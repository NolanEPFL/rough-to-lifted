"""scripts/11_two_factor_value_diagnostics.py — does the symmetric split really make
this a *two*-factor model, and do two blocks buy something one block cannot?

Run AFTER scripts/10 (uses the calibrated JSONs in results/09_symmetric/). Produces
results/09_symmetric/DIAGNOSTICS.md + figures + diagnostics_value.json.

Three questions, answered honestly:

  1. BUG GONE (sharp contrast). To give block 2 the SAME variance share the symmetric
     model assigns it (V0^(2)=w·ξ0(0)), the OLD subtractive model g0^(1)=ξ0-E[V^(2)]
     would clip a large fraction of the maturity window and overshoot E[V]; the additive
     split (1-w)ξ0 + wξ0 clips 0% with E[V]=ξ0 exactly. We quantify both on every date.

  2. IS IT REALLY A TWO-FACTOR / DOES BLOCK 2 ADD VALUE ONE BLOCK CANNOT?
     (a) Nested value: SYM-TF strictly nests the single lifted block (w=0 ≡ LH-geo).
         Report the in-sample loss SYM-TF earns over its own w=0 restriction and over
         the best single-factor baseline.
     (b) Skew-shape freedom (the decisive structural test): a single lifted block has a
         RIGID ATM-skew term-structure shape set by one H (≈ power law T^{H-1/2}); nu,rho
         only rescale it. The two-factor adds a second timescale, accessing skew shapes
         with a long-end "shoulder" that NO single H reproduces. We plot the normalized
         skew shapes of both families and measure the shape residual a single block leaves
         when asked to match a two-factor skew curve.
     (c) Per-maturity skew decomposition at the calibrated params (block-1 vs block-2).
     (d) Timescale separation: calibrated kappa2 vs block-1 geometric-grid speeds.

NEW WORK; no existing/tracked file is modified.
"""
from __future__ import annotations

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
from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.characteristic_function import SymmetricTwoFactorCF
from src.common.cos_method import cos_call_prices, cos_truncation_interval
from src.common.black_scholes import bs_implied_vol

OUTDIR = _ROOT / "results" / "09_symmetric"
FIGDIR = OUTDIR / "figures"
DATES = list(C.DATE_PAIRS.keys())


def load_sym(date, lt, seed=42):
    f = OUTDIR / f"sym_{date}_lam{lt}_s{seed}.json"
    return json.load(open(f)) if f.exists() else None


def _iv_at(cf_callable, fv, S0, K, T, n_steps_unused=None):
    total_var = fv.integrated(T)
    a, b = cos_truncation_interval(T, total_var, 12.0)
    px = np.maximum(cos_call_prices(cf_callable, S0, np.asarray(K, float), T, a, b, 256), 0.0)
    return np.array([bs_implied_vol(float(p), S0, float(k), T, "C") for p, k in zip(px, K)])


def atm_skew_curve(par, fv, S0, T_list, h=0.01, n_steps=400, block=None):
    """FD ATM skew dσ/dk|0 at each T. block=None: full CF; 1: block-1 only; 2: block-2 only."""
    out = {}
    for T in T_list:
        cf = SymmetricTwoFactorCF(par, fv, T, n_steps=n_steps)
        if block is None:
            fn = cf.cf_centered
        elif block == 1:
            fn = cf._cf1.cf_centered
        else:
            fn = cf._cf2.cf_centered
        K = S0 * np.exp(np.array([-h, 0.0, h]))
        iv = _iv_at(fn, fv, S0, K, T)
        if np.all(np.isfinite(iv)):
            out[float(T)] = float((iv[2] - iv[0]) / (2.0 * h))
    return out


# ── Part 1: bug-gone contrast (additive split vs old subtractive, same block-2 share) ──

def part1_bug_gone():
    print("\n=== 1. Bug gone: additive split vs old subtractive at the SAME block-2 share ===")
    print(f"  {'date':11} {'w':>6} {'newClip':>8} {'new|EV/xi0-1|':>14} "
          f"{'oldClip(sameShare)':>18} {'old maxEV/xi0-1':>15}")
    rows = []
    for date in DATES:
        J = load_sym(date, "0", 42)
        if J is None:
            continue
        p = J["params"]; w = p["w"]
        surf, fv, S0 = C.load_surface(date)
        Tmax = float(max(surf.ivs_per_T().keys()))
        tg = np.linspace(1e-4, Tmax, 6000)
        xg = np.array([float(fv(t)) for t in tg])
        xi0_0 = float(fv(0.0))
        # NEW additive split
        g1 = (1 - w) * xg; g2 = w * xg
        new_clip = float(np.mean((g1 < 0) | (g2 < 0)))
        new_relEV = float(np.max(np.abs((g1 + g2) / xg - 1.0)))
        # OLD subtractive model giving block 2 the SAME initial share V2_0=w·xi0(0):
        # constant block-2 mean E[V2]≈w·xi0(0) (theta2=V2_0=w·xi0(0)); g0^(1)=xi0 - w·xi0(0).
        ev2_old = w * xi0_0
        g1_old_raw = xg - ev2_old
        old_clip = float(np.mean(g1_old_raw < 1e-8))
        g1_old = np.maximum(g1_old_raw, 1e-8)
        ev_old = g1_old + ev2_old
        old_relEV = float(np.max(np.abs(ev_old / xg - 1.0)))
        rows.append(dict(date=date, w=w, new_clip=new_clip, new_relEV=new_relEV,
                         old_clip_same_share=old_clip, old_maxEV_dev=old_relEV,
                         xi0_0=xi0_0, xi_min=float(xg.min())))
        print(f"  {date:11} {w:>6.3f} {new_clip:>8.0e} {new_relEV:>14.1e} "
              f"{old_clip:>17.1%} {old_relEV:>+15.1%}")
    return rows


# ── Part 2a: nested value (SYM-TF vs its own w=0 restriction = LH-geo) ─────────────────

def part2a_nested_value():
    print("\n=== 2a. Nested value: SYM-TF vs single-block restriction (w=0 ≡ LH-geo) ===")
    print(f"  {'date':11} {'lam':4} {'SYM ivIS':>9} {'LHgeo ivIS':>11} {'best1F':>8} "
          f"{'Δ vs LHgeo%':>11} {'Δ vs best1F%':>12} {'w':>6}")
    rows = []
    for date in DATES:
        test = C.DATE_PAIRS[date]
        for lt in ("0", "inf"):
            J = load_sym(date, lt, 42)
            oneF = C.single_factor_iv(date, test, lt)
            if J is None or oneF is None:
                continue
            sym = J["in_sample"]["iv_rmse"]
            lhgeo = oneF["all_iv_IS"].get("LH-geo", float("nan"))
            best = oneF["iv_IS"]; w = J["params"]["w"]
            d_lh = 100 * (sym - lhgeo) / lhgeo
            d_best = 100 * (sym - best) / best
            rows.append(dict(date=date, lam=lt, sym_ivIS=sym, lhgeo_ivIS=lhgeo,
                             best1F_ivIS=best, best1F_model=oneF["model"],
                             delta_vs_lhgeo_pct=d_lh, delta_vs_best1F_pct=d_best, w=w))
            print(f"  {date:11} {lt:4} {sym:>9.4f} {lhgeo:>11.4f} {best:>8.4f} "
                  f"{d_lh:>+11.1f} {d_best:>+12.1f} {w:>6.3f}")
    return rows


# ── Part 2b: skew-shape freedom (single rigid H-family vs two-factor shoulder) ─────────

def part2b_skew_shapes(date="2024-12-04"):
    print(f"\n=== 2b. Skew-shape freedom ({date}): single-block H-family vs two-factor ===")
    surf, fv, S0 = C.load_surface(date)
    Ts = np.array(sorted(surf.ivs_per_T().keys()))
    T_grid = np.geomspace(max(Ts[0], 0.03), float(Ts[-1]), 12)
    T0 = float(T_grid[np.argmin(np.abs(T_grid - 0.25))])  # normalisation maturity

    # Single-block family: w=0, vary H1 (nu1,rho1 fixed). Shape = skew(T)/skew(T0).
    single = {}
    for H1 in [0.05, 0.10, 0.20, 0.35, 0.48]:
        par = C.make_params(0.0, H1, 0.7, -0.7, 1.0, 0.5, -0.7)
        sk = atm_skew_curve(par, fv, S0, T_grid)
        if T0 in sk and abs(sk[T0]) > 1e-9:
            single[H1] = {T: sk[T] / sk[T0] for T in sk}

    # Two-factor family: fix H1=0.10, vary (w, kappa2) -> second-timescale shoulder.
    twof = {}
    for (w, k2) in [(0.5, 0.2), (0.5, 1.0), (0.7, 0.3), (0.35, 0.15)]:
        par = C.make_params(w, 0.10, 0.7, -0.7, k2, 0.7, -0.7)
        sk = atm_skew_curve(par, fv, S0, T_grid)
        if T0 in sk and abs(sk[T0]) > 1e-9:
            twof[(w, k2)] = {T: sk[T] / sk[T0] for T in sk}

    # Residual: best single-block H fit to each two-factor normalized shape (shape RMSE).
    def shape_vec(d):
        return np.array([d.get(float(T), np.nan) for T in T_grid])
    resid = {}
    for key, tf in twof.items():
        tfv = shape_vec(tf)
        best = np.inf
        for H1, s in single.items():
            sv = shape_vec(s)
            m = np.isfinite(tfv) & np.isfinite(sv)
            if m.sum() >= 4:
                best = min(best, float(np.sqrt(np.mean((tfv[m] - sv[m]) ** 2))))
        resid[str(key)] = best
        print(f"  two-factor (w={key[0]},κ2={key[1]}): min shape-RMSE vs ANY single-block H = {best:.3f}")

    # Figure
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for H1, s in single.items():
        T = sorted(s); axes[0].plot(T, [s[t] for t in T], "o-", label=f"H={H1:.2f}")
    axes[0].set_title("Single lifted block: rigid 1-parameter (H) skew shape")
    axes[0].set_xscale("log"); axes[0].set_xlabel("T (yr)")
    axes[0].set_ylabel(f"normalized ATM skew  s(T)/s({T0:.2f})")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for key, s in twof.items():
        T = sorted(s); axes[1].plot(T, [s[t] for t in T], "s-", label=f"w={key[0]},κ2={key[1]}")
    # overlay the single-block envelope (grey) for contrast
    for H1, s in single.items():
        T = sorted(s); axes[1].plot(T, [s[t] for t in T], "-", color="0.8", lw=1, zorder=0)
    axes[1].set_title("Two-factor: second timescale adds a long-end shoulder")
    axes[1].set_xscale("log"); axes[1].set_xlabel("T (yr)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"ATM-skew term-structure SHAPE freedom — {date} "
                 "(grey = single-block family)")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"skew_shape_freedom_{date}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIGDIR / f'skew_shape_freedom_{date}.png'}")
    return dict(single_block_H=list(single.keys()),
                two_factor_shape_residual_vs_best_single=resid, T0=T0)


# ── Part 2c: per-maturity skew decomposition at calibrated params ──────────────────────

def part2c_decomposition(date="2024-12-04"):
    print(f"\n=== 2c. Per-maturity skew decomposition at calibrated params ({date}, λ=0) ===")
    J = load_sym(date, "0", 42)
    if J is None:
        return None
    p = J["params"]
    par = C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"])
    # block 2 OFF: nu2 → tiny (kills its stochastic-vol skew, keeps its variance share)
    par_off = C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], 1e-8, p["rho2"])
    surf, fv, S0 = C.load_surface(date)
    Ts = np.array(sorted(surf.ivs_per_T().keys()))
    T_grid = np.geomspace(max(Ts[0], 0.03), float(Ts[-1]), 12)
    sk_full = atm_skew_curve(par, fv, S0, T_grid)
    sk_off = atm_skew_curve(par_off, fv, S0, T_grid)
    common = sorted(set(sk_full) & set(sk_off))
    incr = {T: sk_full[T] - sk_off[T] for T in common}
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(common, [sk_full[T] for T in common], "r^-", label="full two-factor")
    ax.plot(common, [sk_off[T] for T in common], "b.-", label="block 2 off (block-1 skew)")
    ax.plot(common, [incr[T] for T in common], "g--", label="block-2 increment")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("T (yr)"); ax.set_ylabel("ATM skew dσ/dk|0")
    ax.set_title(f"Skew decomposition — {date} (w={p['w']:.3f}, κ2={p['kappa2']:.2f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"skew_decomposition_{date}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  block-2 skew increment: max={max(abs(v) for v in incr.values()):.4f} "
          f"at T≈{max(incr, key=lambda t: abs(incr[t])):.2f}")
    print(f"  saved {FIGDIR / f'skew_decomposition_{date}.png'}")
    return dict(T=common, full=[sk_full[T] for T in common],
                block1=[sk_off[T] for T in common], block2_incr=[incr[T] for T in common])


# ── Part 2d: timescale separation ──────────────────────────────────────────────────────

def part2d_timescales():
    print("\n=== 2d. Timescale separation: kappa2 vs block-1 geometric-grid speeds ===")
    rows = []
    for date in DATES:
        for lt in ("0", "inf"):
            J = load_sym(date, lt, 42)
            if J is None:
                continue
            p = J["params"]
            par = C.make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"])
            x1 = par.x1  # block-1 speeds
            k2 = p["kappa2"]
            # block-2 timescale 1/k2 vs block-1 fastest/slowest timescales 1/x
            slow_b1 = 1.0 / x1.min()  # slowest block-1 mode
            fast_b1 = 1.0 / x1.max()
            tau2 = 1.0 / k2
            sep = tau2 / slow_b1  # >1 => block 2 genuinely slower than block-1's slowest mode
            rows.append(dict(date=date, lam=lt, kappa2=k2, tau2=tau2,
                             b1_slow_tau=slow_b1, b1_fast_tau=fast_b1, sep_ratio=sep, w=p["w"]))
            print(f"  {date} {lt:4}: κ2={k2:6.3f} τ2={tau2:6.3f}yr  "
                  f"block1 τ∈[{fast_b1:.1e},{slow_b1:.2f}]yr  τ2/τ1_slow={sep:.2f}  w={p['w']:.3f}")
    return rows


def main():
    print("=" * 76)
    print("Two-factor value diagnostics (symmetric-split, §8.9.7 remedy)")
    print("=" * 76)
    out = {}
    out["bug_gone"] = part1_bug_gone()
    out["nested_value"] = part2a_nested_value()
    out["skew_shapes"] = part2b_skew_shapes("2024-12-04")
    out["decomposition"] = part2c_decomposition("2024-12-04")
    out["timescales"] = part2d_timescales()
    json.dump(out, open(OUTDIR / "diagnostics_value.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUTDIR / 'diagnostics_value.json'}")


if __name__ == "__main__":
    main()
