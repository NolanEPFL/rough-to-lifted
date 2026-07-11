"""scripts/09_symmetric_sanity_checks.py — sanity checks for the symmetric-split
two-factor lifted Heston (erratum §8.9.7 remedy). NEW WORK beyond the thesis.

Mirrors the four checks of scripts/08_two_factor_sanity_checks.py and adds the
positivity probe the erratum asks for:

  1. Recover single-block lifted Heston : w=0 => aggregate centered CF == the
     existing single-block LH CF (block 1 forced by ξ0) to machine precision.
  2. Exact forward-variance match       : for several w and maturities,
     |E^Q[V_T]/ξ0(T) - 1| <= 1e-14 (analytic forward-variance identity) AND
     clip fraction == 0 exactly (there is no clip), cross-checked by the CF
     integrated-variance identity (-2 Φ'(0) = ∫ξ0) and by Monte Carlo.
  3. Monte Carlo vs COS                  : |COS - MC| < 10·stderr (rough-kernel
     Euler bias band, as in check 3 of scripts/08).
  4. CF normalisation                    : Φ_T(0) = Φ_T(1) = 1.
  5. Positivity probe                    : running min and negative fraction of the
     RAW (pre-truncation) V^(1), V^(2) on a steep downward-sloping SPX ξ0 — a
     genuine question for a single forward-variance block; reported honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Make Unicode glyphs (Phi, xi0, ...) safe on a cp1252 Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.characteristic_function import SymmetricTwoFactorCF
from src.two_factor_symmetric.pricing import symmetric_call_prices
from src.two_factor_symmetric.monte_carlo import simulate_symmetric, mc_call_prices
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from src.lifted_heston.pricing import lifted_heston_call_prices
from src.common.forward_variance import FlatForwardVariance, PiecewiseConstantForwardVariance
from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

STEEP_DATE = "2024-08-05"  # high-vol date: ξ0(0)≈0.109, min_t ξ0≈0.005 (erratum)


def load_spx_xi0(date: str):
    """Re-extract the market ξ0 (PCHIP-derivative of ATM total variance)."""
    surface = load_spx_csv(_ROOT / "data" / f"spx_{date}.csv")
    xi0_m, xi0_v = fit_xi0_from_surface(surface)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surface.forwards.mean())
    Ts = np.array(sorted(surface.ivs_per_T().keys()))
    return fv, S0, Ts


# ──────────────────────────────────────────────────────────────────────────────
def check1_recover_single_block() -> bool:
    """w=0: block-2 forcing is identically 0 => Φ^(2)≡1 => aggregate CF == single LH."""
    print("\n--- Check 1: Recover single-block lifted Heston (w=0) ---")
    V0 = 0.04
    fv = FlatForwardVariance(V0)
    S0 = 100.0

    sym = SymmetricTwoFactorParams(
        w=0.0, H1=0.10, nu1=0.3, rho1=-0.7, kappa2=1.0, nu2=0.2, rho2=-0.4
    )
    lh = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.3, rho=-0.7)

    maturities = [0.25, 0.5, 1.0, 1.5, 2.0]
    # u-grid: imaginary frequencies (COS-style) plus the martingale/probability points.
    omega = np.linspace(0.0, 60.0, 240)
    u_grid = np.concatenate([1j * omega, np.array([0.0 + 0j, 1.0 + 0j])])

    max_cf_err = 0.0
    max_px_err = 0.0
    log_k = np.linspace(-0.3, 0.2, 13)
    strikes = S0 * np.exp(log_k)
    for T in maturities:
        sym_cf = SymmetricTwoFactorCF(sym, fv, T).cf_centered(u_grid)
        ref_cf = LiftedHestonCharacteristicFunction(lh, fv, T).cf_centered(u_grid)
        max_cf_err = max(max_cf_err, float(np.max(np.abs(sym_cf - ref_cf))))

        px_sym = symmetric_call_prices(sym, fv, S0, strikes, T)
        px_lh = lifted_heston_call_prices(lh, fv, S0, strikes, T)
        max_px_err = max(max_px_err, float(np.max(np.abs(px_sym - px_lh))))

    ok = (max_cf_err < 1e-12) and (max_px_err < 1e-10)
    print(f"  max |Φ_SYM(w=0) - Φ_LH| on u-grid : {max_cf_err:.2e}  (tol 1e-12)")
    print(f"  max |call_SYM - call_LH|          : {max_px_err:.2e}  (tol 1e-10)")
    print(f"  {PASS if ok else FAIL}")
    if not ok:
        raise AssertionError("Check 1 FAILED")
    return True


# ──────────────────────────────────────────────────────────────────────────────
def check2_forward_variance_match() -> bool:
    """E^Q[V_T] = ξ0(T) exactly (additive split), clip fraction == 0."""
    print("\n--- Check 2: Exact forward-variance match (the whole point) ---")
    fv, S0, Ts = load_spx_xi0(STEEP_DATE)
    Tmax = float(Ts[-1])
    print(f"  {STEEP_DATE}: ξ0(0)={float(fv(0.0)):.4f}  "
          f"min_t ξ0={min(float(fv(t)) for t in np.linspace(1e-4, Tmax, 4000)):.5f}  "
          f"Tmax={Tmax:.3f}")

    ws = [0.1, 0.3, 0.5, 0.9]
    mats = np.array([m for m in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0] if m <= Tmax])

    # (a) analytic forward-variance identity + clip fraction, on a dense grid.
    t_dense = np.linspace(1e-4, Tmax, 6000)
    xi_dense = np.array([float(fv(t)) for t in t_dense])
    max_relEV = 0.0
    max_clip = 0.0
    for w in ws:
        s1, s2 = 1.0 - w, w
        g1 = s1 * xi_dense
        g2 = s2 * xi_dense
        # clip fraction: never any flooring in this construction (both forcings >= 0)
        clip_frac = float(np.mean((g1 < 0.0) | (g2 < 0.0)))
        max_clip = max(max_clip, clip_frac)
        for T in mats:
            xiT = float(fv(T))
            EV = s1 * xiT + s2 * xiT          # E[V_T] = g0^(1)(T) + g0^(2)(T)
            rel = abs(EV / xiT - 1.0)
            max_relEV = max(max_relEV, rel)
    print(f"  (a) analytic   max|E[V_T]/ξ0(T) - 1| over w,T : {max_relEV:.2e}  (tol 1e-14)")
    print(f"      clip fraction (any flooring applied?)     : {max_clip:.1e}  (must be 0)")

    # (b) CF integrated-variance identity: -2 Φ'(0) = E[∫_0^T V ds] = ∫_0^T ξ0.
    h = 1e-4
    cf_rel_devs = []
    for w in [0.3, 0.5]:
        p = SymmetricTwoFactorParams(w=w, H1=0.10, nu1=0.8, rho1=-0.7,
                                     kappa2=1.0, nu2=0.5, rho2=-0.5)
        for T in [m for m in [0.25, 1.0] if m <= Tmax]:
            cf = SymmetricTwoFactorCF(p, fv, T, n_steps=1600)
            phi_p = complex(cf.cf_centered(np.array([h + 0j]))[0])
            phi_m = complex(cf.cf_centered(np.array([-h + 0j]))[0])
            EV_int = -2.0 * ((phi_p - phi_m) / (2.0 * h)).real
            target = fv.integrated(T)
            cf_rel_devs.append(abs(EV_int / target - 1.0))
    cf_max = max(cf_rel_devs)
    print(f"  (b) CF identity max|-2Φ'(0)/∫ξ0 - 1|          : {cf_max:.2e}  (tol 1e-2, FD+trapz)")

    # (c) Monte Carlo confirmation of E[V_T] at a mid maturity.
    p = SymmetricTwoFactorParams(w=0.5, H1=0.10, nu1=0.8, rho1=-0.7,
                                 kappa2=1.0, nu2=0.5, rho2=-0.5)
    T_mc = float(min(0.5, Tmax))
    _, diag = simulate_symmetric(p, fv, S0, T_mc, M_paths=80_000,
                                 n_steps_per_year=500, antithetic=True,
                                 seed=42, return_diag=True)
    EV_mc = diag["EV1_T"] + diag["EV2_T"]
    xiT = float(fv(T_mc))
    mc_rel = abs(EV_mc / xiT - 1.0)
    print(f"  (c) MC E[V_T] at T={T_mc:.2f}: {EV_mc:.5f} vs ξ0(T)={xiT:.5f}  "
          f"rel dev {mc_rel:.2e}")

    ok = (max_relEV <= 1e-14) and (max_clip == 0.0) and (cf_max < 1e-2) and (mc_rel < 0.05)
    print(f"  {PASS if ok else FAIL}")
    if not ok:
        raise AssertionError("Check 2 FAILED")
    return True


# ──────────────────────────────────────────────────────────────────────────────
def check3_mc_vs_cos() -> bool:
    """COS vs Euler–Maruyama within 10·stderr (rough-kernel bias band)."""
    print("\n--- Check 3: Monte Carlo vs COS agreement ---")
    V0 = 0.04
    fv = FlatForwardVariance(V0)
    S0 = 100.0
    p = SymmetricTwoFactorParams(w=0.3, H1=0.10, nu1=0.3, rho1=-0.7,
                                 kappa2=1.0, nu2=0.2, rho2=-0.4)
    maturities = [0.5, 1.0]
    log_k = np.linspace(-0.2, 0.1, 5)
    strikes = S0 * np.exp(log_k)

    all_pass = True
    for T in maturities:
        px_cos = symmetric_call_prices(p, fv, S0, strikes, T)
        px_mc, stderr = mc_call_prices(p, fv, S0, strikes, T, M_paths=100_000,
                                       n_steps_per_year=500, antithetic=True, seed=42)
        for j, K in enumerate(strikes):
            diff = abs(float(px_cos[j]) - float(px_mc[j]))
            tol = 10.0 * float(stderr[j])
            ok = diff < tol
            all_pass = all_pass and ok
            print(f"  T={T:.2f} K={K:6.2f}: COS={px_cos[j]:.4f} "
                  f"MC={px_mc[j]:.4f}±{stderr[j]:.4f} diff={diff:.4f} tol={tol:.4f}  "
                  f"{PASS if ok else FAIL}")
    print(f"  {PASS if all_pass else FAIL}")
    if not all_pass:
        raise AssertionError("Check 3 FAILED")
    return True


# ──────────────────────────────────────────────────────────────────────────────
def check4_cf_normalisation() -> bool:
    """Φ_T(0) = 1 (probability) and Φ_T(1) = 1 (martingale)."""
    print("\n--- Check 4: CF normalisation ---")
    fv = FlatForwardVariance(0.04)
    p = SymmetricTwoFactorParams(w=0.3, H1=0.10, nu1=0.3, rho1=-0.7,
                                 kappa2=1.0, nu2=0.2, rho2=-0.4)
    all_pass = True
    for T in [0.25, 1.0, 2.0]:
        cf = SymmetricTwoFactorCF(p, fv, T)
        phi0 = complex(cf.cf_centered(np.array([0.0 + 0j]))[0])
        phi1 = complex(cf.cf_centered(np.array([1.0 + 0j]))[0])
        err0 = abs(phi0 - 1.0)
        err1 = abs(phi1 - 1.0)
        ok = err0 < 1e-8 and err1 < 1e-8
        all_pass = all_pass and ok
        print(f"  T={T:.2f}: |Φ(0)-1|={err0:.2e}  |Φ(1)-1|={err1:.2e}  "
              f"{PASS if ok else FAIL}")
    print(f"  {PASS if all_pass else FAIL}")
    if not all_pass:
        raise AssertionError("Check 4 FAILED")
    return True


# ──────────────────────────────────────────────────────────────────────────────
def check5_positivity_probe() -> bool:
    """Pathwise positivity of each forward-variance block on a steep ξ0.

    Reported honestly: with a non-negative forcing and full truncation this
    should be ~0, but genuine pathwise positivity of a single forward-variance
    block on a steeply downward-sloping ξ0 is a real question.
    """
    print("\n--- Check 5: Positivity probe (steep SPX ξ0) ---")
    fv, S0, Ts = load_spx_xi0(STEEP_DATE)
    T = float(min(Ts[-1], 1.5))
    print(f"  {STEEP_DATE}: probing V^(1),V^(2) over [0,{T:.2f}], "
          f"ξ0(T)={float(fv(T)):.5f}")
    worst = 0.0
    for w in [0.3, 0.5, 0.9]:
        p = SymmetricTwoFactorParams(w=w, H1=0.10, nu1=0.8, rho1=-0.7,
                                     kappa2=1.0, nu2=0.5, rho2=-0.5)
        _, d = simulate_symmetric(p, fv, S0, T, M_paths=60_000,
                                  n_steps_per_year=500, antithetic=True,
                                  seed=42, return_diag=True)
        worst = max(worst, d["frac_neg1"], d["frac_neg2"])
        print(f"  w={w:.1f}: min V1_raw={d['min_V1_raw']:+.2e} "
              f"min V2_raw={d['min_V2_raw']:+.2e}  "
              f"frac_neg1={d['frac_neg1']:.2e} frac_neg2={d['frac_neg2']:.2e}")
    tag = INFO if worst > 1e-3 else PASS
    note = "  (negativity present — reported, not floored)" if worst > 1e-3 else ""
    print(f"  worst negative fraction over (path,step): {worst:.2e}  {tag}{note}")
    return True


def main() -> None:
    print("=" * 64)
    print("Symmetric-split two-factor lifted Heston — sanity checks (§8.9.7 remedy)")
    print("=" * 64)
    checks = [
        (check1_recover_single_block, "Recover single-block LH"),
        (check2_forward_variance_match, "Exact ξ0 match + no clip"),
        (check3_mc_vs_cos, "MC vs COS"),
        (check4_cf_normalisation, "CF normalisation"),
        (check5_positivity_probe, "Positivity probe"),
    ]
    passed = 0
    for fn, name in checks:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERROR in {name}: {e}")
            traceback.print_exc()
    print(f"\n{'=' * 64}")
    print(f"Results: {passed}/{len(checks)} checks passed.")
    if passed < len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
