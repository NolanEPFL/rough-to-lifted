"""Investigation of the check-5 positivity finding for the symmetric-split model.

Asked by the user before the fair run. Answers four questions about the raw
(pre-truncation) negativity of a single forward-variance lifted-Heston block:

  A. How does it scale with vol-of-vol nu1 and roughness H1?
  B. Is it a time-discretization artifact (does it vanish as dt -> 0) or an
     intrinsic continuous-time property of the finite-n affine representation?
  C. The PRICING-relevant quantity: the truncated variance MASS
     E[int V^- dt]/E[int V^+ dt] (pointwise frac_neg overstates the impact if
     the negative dips are shallow/brief), and whether COS (exact affine) and MC
     (full-truncation) implied vols actually diverge on a steep curve.
  D. Correctness: an INDEPENDENT re-implementation of the single-block exact-OU
     simulator, cross-checked (i) against the package simulator's frac_neg and
     (ii) against the validated existing two-factor MC (degenerate block 2) and
     the single-block lifted-Heston COS prices.

Writes results/09_symmetric/diagnostics/{positivity.json, positivity.png}.
NEW WORK; no existing/tracked file is modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lifted_heston.params import geometric_grid, LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_call_prices, lifted_heston_iv_surface
from src.common.forward_variance import FlatForwardVariance, PiecewiseConstantForwardVariance
from src.common.black_scholes import bs_implied_vol
from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface

from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.pricing import symmetric_call_prices, symmetric_iv_surface
from src.two_factor_symmetric.monte_carlo import simulate_symmetric

# Existing (validated) two-factor MC, used only as an external cross-check.
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.monte_carlo import simulate_two_factor

OUT = _ROOT / "results" / "09_symmetric" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
STEEP_DATE = "2024-08-05"


def load_spx_xi0(date):
    surface = load_spx_csv(_ROOT / "data" / f"spx_{date}.csv")
    xi0_m, xi0_v = fit_xi0_from_surface(surface)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surface.forwards.mean())
    Ts = np.array(sorted(surface.ivs_per_T().keys()))
    return fv, S0, Ts


def probe_block(c, x, nu, rho, fv, scale, T, M, n_spy, seed=42):
    """Independent single-block exact-OU simulator with full positivity diagnostics.

    Simulates V_raw = scale*xi0(t) + sum_i c_i U_i with dU_i = -x_i U_i dt +
    nu sqrt(max(V_raw,0)) dW (full truncation in the diffusion). Returns pointwise
    negativity AND the truncated-mass ratio E[int V^- dt]/E[int V^+ dt].
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(round(T * n_spy)))
    dt = T / n
    sqdt = np.sqrt(dt)
    a = np.exp(-x * dt)
    beta_sq = (1.0 - np.exp(-2.0 * x * dt)) / (2.0 * x)
    cov = (1.0 - a) / (x * dt)
    sres = np.sqrt(np.maximum(beta_sq - cov ** 2 * dt, 0.0))
    U = np.zeros((M, len(c)))
    min_v = np.inf
    neg = 0
    npairs = 0
    negmass = 0.0
    posmass = 0.0
    ev_T = np.nan
    t = 0.0
    for _ in range(n):
        dW = rng.standard_normal((M,)) * sqdt
        eps = rng.standard_normal((M, len(c)))
        g = scale * float(fv(t))
        V_raw = g + U @ c
        min_v = min(min_v, float(V_raw.min()))
        neg += int(np.count_nonzero(V_raw < 0.0))
        npairs += M
        V_pos = np.maximum(V_raw, 0.0)
        V_neg = np.maximum(-V_raw, 0.0)
        posmass += float(V_pos.sum()) * dt
        negmass += float(V_neg.sum()) * dt
        sv = nu * np.sqrt(V_pos)
        U = U * a[None, :] + sv[:, None] * (cov[None, :] * dW[:, None] + sres[None, :] * eps)
        t += dt
    ev_T = float((scale * float(fv(T)) + U @ c).mean())
    return dict(
        frac_neg=neg / npairs,
        min_v=min_v,
        negmass_ratio=(negmass / posmass) if posmass > 0 else np.nan,
        ev_T=ev_T,
        ev_target=scale * float(fv(T)),
    )


def section_A(fv_steep):
    print("\n=== A. Negativity vs vol-of-vol (nu1) and roughness (H1) ===")
    print("    block 1 alone (w=0, forced by full xi0), steep 2024-08-05, T=1.5, 500/yr")
    print(f"    {'H1':>5} {'nu1':>5} {'frac_neg':>9} {'min_Vraw':>10} {'negmass/posmass':>16}")
    grid = {}
    for H1 in [0.05, 0.10, 0.20, 0.30]:
        c, x = geometric_grid(H1, 20, 2.5)
        for nu1 in [0.2, 0.4, 0.8, 1.5]:
            r = probe_block(c, x, nu1, -0.7, fv_steep, 1.0, 1.5, 40_000, 500)
            grid[(H1, nu1)] = r
            print(f"    {H1:>5.2f} {nu1:>5.2f} {r['frac_neg']:>9.2%} "
                  f"{r['min_v']:>+10.3f} {r['negmass_ratio']:>16.2e}")
    return grid


def section_B(fv_steep):
    print("\n=== B. Discretization convergence (does it vanish as dt -> 0?) ===")
    print("    H1=0.10, nu1=0.8, rho1=-0.7, steep curve, T=1.5")
    print(f"    {'steps/yr':>9} {'frac_neg':>9} {'min_Vraw':>10} {'negmass/posmass':>16} {'E[V_T]/tgt-1':>13}")
    c, x = geometric_grid(0.10, 20, 2.5)
    out = {}
    for n_spy in [250, 500, 1000, 2000, 4000]:
        r = probe_block(c, x, 0.8, -0.7, fv_steep, 1.0, 1.5, 40_000, n_spy)
        out[n_spy] = r
        reldev = r["ev_T"] / r["ev_target"] - 1.0
        print(f"    {n_spy:>9} {r['frac_neg']:>9.2%} {r['min_v']:>+10.3f} "
              f"{r['negmass_ratio']:>16.2e} {reldev:>+13.2e}")
    return out


def section_C(fv_steep, S0_steep, Ts_steep):
    print("\n=== C. Does the negativity bias COS pricing? COS (exact affine) vs MC (truncated) IV ===")
    print("    steep curve, moderate params w=0.5 nu1=0.4 nu2=0.4 H1=0.10, seed 42")
    p = SymmetricTwoFactorParams(w=0.5, H1=0.10, nu1=0.4, rho1=-0.7,
                                 kappa2=1.0, nu2=0.4, rho2=-0.5)
    mats = [float(m) for m in [0.1, 0.25, 0.5, 1.0] if m <= float(Ts_steep[-1])]
    rows = []
    print(f"    {'T':>5} {'k=logK/S0':>10} {'IV_COS':>8} {'IV_MC':>8} {'diff(volpts)':>12}")
    for T in mats:
        log_k = np.array([-0.10, -0.05, 0.0, 0.05])
        strikes = S0_steep * np.exp(log_k)
        # COS IVs
        px_cos = symmetric_call_prices(p, fv_steep, S0_steep, strikes, T, n_steps=1600)
        iv_cos = np.array([bs_implied_vol(float(pp), S0_steep, float(K), T, "C")
                           for pp, K in zip(px_cos, strikes)])
        # MC IVs (truncated paths)
        from src.two_factor_symmetric.monte_carlo import mc_call_prices
        px_mc, _ = mc_call_prices(p, fv_steep, S0_steep, strikes, T,
                                  M_paths=200_000, n_steps_per_year=1000,
                                  antithetic=True, seed=7)
        iv_mc = np.array([bs_implied_vol(float(pp), S0_steep, float(K), T, "C")
                          for pp, K in zip(px_mc, strikes)])
        for kk, a, b in zip(log_k, iv_cos, iv_mc):
            d = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
            rows.append(dict(T=T, k=float(kk), iv_cos=float(a), iv_mc=float(b), diff=float(d)))
            print(f"    {T:>5.2f} {kk:>10.3f} {a:>8.4f} {b:>8.4f} {d:>12.4f}")
    diffs = np.array([r["diff"] for r in rows if np.isfinite(r["diff"])])
    print(f"    -> max |IV_COS - IV_MC| = {np.nanmax(np.abs(diffs)):.4f} vol pts "
          f"(MC has rough-kernel Euler bias of its own)")
    return rows


def section_D(fv_steep, S0_steep):
    print("\n=== D. Correctness cross-checks ===")
    # (i) independent probe vs package simulator: frac_neg at w=0 must agree.
    c, x = geometric_grid(0.10, 20, 2.5)
    indep = probe_block(c, x, 0.8, -0.7, fv_steep, 1.0, 1.5, 40_000, 500)
    p_w0 = SymmetricTwoFactorParams(w=0.0, H1=0.10, nu1=0.8, rho1=-0.7,
                                    kappa2=1.0, nu2=0.5, rho2=-0.5)
    _, dpkg = simulate_symmetric(p_w0, fv_steep, S0_steep, 1.5, M_paths=40_000,
                                 n_steps_per_year=500, antithetic=False, seed=42,
                                 return_diag=True)
    print(f"  (i) frac_neg(block1): independent probe={indep['frac_neg']:.4f}  "
          f"package sim={dpkg['frac_neg1']:.4f}  "
          f"|diff|={abs(indep['frac_neg']-dpkg['frac_neg1']):.4f}")

    # (ii) price match: sym(w=0) vs single-block LH vs existing two-factor (block 2 off).
    strikes = S0_steep * np.exp(np.linspace(-0.2, 0.1, 7))
    T = 0.5
    px_sym = symmetric_call_prices(p_w0, fv_steep, S0_steep, strikes, T, n_steps=200)
    lh = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.8, rho=-0.7)
    px_lh = lifted_heston_call_prices(lh, fv_steep, S0_steep, strikes, T, n_steps=200)
    # existing two-factor with degenerate block 2 (theta2=V2_0=nu2=0) -> block1 forced by full xi0
    from src.two_factor_lifted_heston.pricing import two_factor_lh_call_prices
    p_tf = TwoFactorLiftedHestonParams(H1=0.10, n1=20, r_n1=2.5, nu1=0.8, rho1=-0.7,
                                       lam2=1.0, theta2=0.0, nu2=1e-12, rho2=-0.5, V2_0=0.0)
    px_tf = two_factor_lh_call_prices(p_tf, fv_steep, S0_steep, strikes, T, n_steps=200)
    e_lh = float(np.max(np.abs(px_sym - px_lh)))
    e_tf = float(np.max(np.abs(px_sym - px_tf)))
    print(f"  (ii) max|call_SYM(w=0) - call_LH|              = {e_lh:.2e}")
    print(f"       max|call_SYM(w=0) - call_existingTF(b2off)| = {e_tf:.2e}")

    # (iii) MC moment match vs existing two-factor MC (block 2 off), same paths/steps.
    lr_sym = simulate_symmetric(p_w0, fv_steep, S0_steep, 1.0, M_paths=60_000,
                                n_steps_per_year=500, antithetic=True, seed=11)
    lr_tf = simulate_two_factor(p_tf, fv_steep, S0_steep, 1.0, M_paths=60_000,
                                n_steps_per_year=500, antithetic=True, seed=11)
    print(f"  (iii) MC log-return moments (T=1, block1 only): "
          f"mean SYM={lr_sym.mean():+.5f} TF={lr_tf.mean():+.5f} ; "
          f"std SYM={lr_sym.std():.5f} TF={lr_tf.std():.5f}")
    return dict(frac_neg_indep=indep["frac_neg"], frac_neg_pkg=dpkg["frac_neg1"],
                price_err_lh=e_lh, price_err_tf=e_tf,
                mean_sym=float(lr_sym.mean()), mean_tf=float(lr_tf.mean()),
                std_sym=float(lr_sym.std()), std_tf=float(lr_tf.std()))


def make_figure(gridA, outB):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # Panel 1: negmass ratio vs nu1, one line per H1
    ax = axes[0]
    H1s = sorted(set(h for (h, _) in gridA))
    nu1s = sorted(set(n for (_, n) in gridA))
    for H1 in H1s:
        y = [gridA[(H1, nu1)]["negmass_ratio"] for nu1 in nu1s]
        ax.plot(nu1s, y, "o-", label=f"H1={H1:.2f}")
    ax.set_xlabel(r"vol-of-vol $\nu_1$")
    ax.set_ylabel(r"truncated mass  $E[\int V^- dt]/E[\int V^+ dt]$")
    ax.set_title("A. Truncated variance mass vs $\\nu_1$, $H_1$ (steep $\\xi_0$)")
    ax.legend(); ax.grid(alpha=0.3)
    # Panel 2: convergence in steps/yr
    ax = axes[1]
    steps = sorted(outB)
    ax.plot(steps, [outB[s]["frac_neg"] for s in steps], "s-", label="pointwise frac_neg")
    ax.plot(steps, [outB[s]["negmass_ratio"] for s in steps], "^-", label="truncated mass ratio")
    ax.set_xscale("log")
    ax.set_xlabel("Euler steps / year")
    ax.set_ylabel("fraction / ratio")
    ax.set_title("B. Step-size convergence (H1=0.10, $\\nu_1$=0.8)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "positivity.png", dpi=130)
    print(f"\nSaved figure -> {OUT / 'positivity.png'}")


def main():
    print("=" * 72)
    print("Symmetric-split positivity investigation (pre-fair-run)")
    print("=" * 72)
    fv, S0, Ts = load_spx_xi0(STEEP_DATE)
    gridA = section_A(fv)
    outB = section_B(fv)
    rowsC = section_C(fv, S0, Ts)
    outD = section_D(fv, S0)
    make_figure(gridA, outB)

    payload = {
        "A_nu_H_sweep": {f"H{h}_nu{n}": v for (h, n), v in gridA.items()},
        "B_step_convergence": {str(k): v for k, v in outB.items()},
        "C_cos_vs_mc_iv": rowsC,
        "D_correctness": outD,
    }
    json.dump(payload, open(OUT / "positivity.json", "w"), indent=2, default=float)
    print(f"Saved data    -> {OUT / 'positivity.json'}")


if __name__ == "__main__":
    main()
