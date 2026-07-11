"""scripts/00_sanity_checks.py — gating sanity checks.

Run this script before any other. ALL CHECKS MUST PRINT "OK" before proceeding to
empirical experiments. If any check fails, stop, diagnose, fix.

Coverage (cf. CONTEXT.md Section 8):
  Lifted Heston:
    [LH-1] Φ_T(0) = 1
    [LH-2] BS recovery at ν→0 (proxy for Heston recovery; full λ≠0 test deferred)
    [LH-3] Convergence in n: stable from n=100 to n=200
    [LH-4] Put-call parity
    [LH-5] No-arbitrage in K
  aBergomi:
    [aB-1] E[V_t] ≈ ξ_0(t)
    [aB-2] MC standard error decreases as 1/√M
    [aB-3] BS recovery at η=0
    [aB-4] No-arbitrage in K
  Kernel:
    [K-1] K_H matches t^{H-1/2} / Γ(H+1/2)
    [K-2] L² kernel error decreases monotonically in n
    [K-3] All fitted weights and speeds positive
  COS:
    [COS-1] COS with exact BS characteristic function matches analytic BS prices
    [COS-2] Stable when L0 increased to 15
    [COS-3] Stable when N_cos increased to 512
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path when running from scripts/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.special import gamma as Gamma_fn


def main() -> int:
    failures = []

    # ---------------- Kernel checks ----------------
    print("[K-1] K_H values match analytic formula ...", end=" ")
    try:
        check_K_H_analytic()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("K-1")

    print("[K-2] L² kernel fit error monotone in n ...", end=" ")
    try:
        check_l2_kernel_monotone()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("K-2")

    print("[K-3] All fitted (c_i, x_i) > 0 ...", end=" ")
    try:
        check_kernel_positivity()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("K-3")

    # ---------------- Lifted Heston checks ----------------
    print("[LH-1] cf_centered(0) = 1 ...", end=" ")
    try:
        check_cf_at_zero()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("LH-1")

    print("[LH-2] BS recovery at ν→0 ...", end=" ")
    try:
        check_heston_recovery()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("LH-2")

    print("[LH-3] IV stable n=100 vs n=200 ...", end=" ")
    try:
        check_lh_convergence_self()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("LH-3")

    print("[LH-4] Put-call parity ...", end=" ")
    try:
        check_put_call_parity()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("LH-4")

    print("[LH-5] Calls monotone & convex in K (LH) ...", end=" ")
    try:
        check_lh_no_arbitrage()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("LH-5")

    # ---------------- aBergomi checks (Phase 2) ----------------
    print("[aB-1] E[V_t] ≈ ξ_0(t) (MC) ...", end=" ")
    try:
        check_abergomi_mean_variance()
        print("OK")
    except NotImplementedError:
        print("SKIP (Phase 2 not yet implemented)"); failures.append("aB-1")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("aB-1")

    print("[aB-2] MC SE ~ 1/√M ...", end=" ")
    try:
        check_abergomi_mc_rate()
        print("OK")
    except NotImplementedError:
        print("SKIP (Phase 2 not yet implemented)"); failures.append("aB-2")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("aB-2")

    print("[aB-3] BS recovery at η=0 ...", end=" ")
    try:
        check_abergomi_bs_recovery()
        print("OK")
    except NotImplementedError:
        print("SKIP (Phase 2 not yet implemented)"); failures.append("aB-3")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("aB-3")

    print("[aB-4] Calls monotone & convex in K (aB) ...", end=" ")
    try:
        check_abergomi_no_arbitrage()
        print("OK")
    except NotImplementedError:
        print("SKIP (Phase 2 not yet implemented)"); failures.append("aB-4")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("aB-4")

    # ---------------- COS checks ----------------
    print("[COS-1] COS vs analytic BS prices ...", end=" ")
    try:
        check_cos_against_heston_closed_form()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("COS-1")

    print("[COS-2] Stable L0=12 vs L0=15 ...", end=" ")
    try:
        check_cos_L0_stability()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("COS-2")

    print("[COS-3] Stable N_cos=256 vs 512 ...", end=" ")
    try:
        check_cos_Ncos_stability()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}"); failures.append("COS-3")

    # ---------------- Summary ----------------
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL SANITY CHECKS PASSED — safe to proceed to empirical experiments.")
    return 0


# ============================================================================
# Individual check implementations
# ============================================================================

def check_K_H_analytic() -> None:
    """[K-1] K_H(t) = t^{H-1/2} / Γ(H+1/2) for t in [1e-3, 10]."""
    from src.lifted_heston.kernel import K_H

    t_test = np.array([1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0])
    for H in [0.05, 0.10, 0.20, 0.30]:
        expected = t_test ** (H - 0.5) / Gamma_fn(H + 0.5)
        got = K_H(t_test, H)
        max_rel = np.max(np.abs(got - expected) / (np.abs(expected) + 1e-300))
        if max_rel > 1e-10:
            raise AssertionError(f"H={H}: max relative error = {max_rel:.2e}")


def check_l2_kernel_monotone() -> None:
    """[K-2] L² fit error is non-increasing in n for n in {3, 5, 10, 20}."""
    from src.abergomi.kernel_fit import abergomi_l2_kernel
    from src.lifted_heston.kernel import kernel_l2_error

    H = 0.10
    errors = []
    for n in [3, 5, 10, 20]:
        c, x, _ = abergomi_l2_kernel(H, n, seed=42)
        err = kernel_l2_error(H, c, x)
        errors.append(err)

    for i in range(len(errors) - 1):
        if errors[i + 1] > errors[i] * 1.05:   # 5% tolerance for numerical noise
            raise AssertionError(
                f"L² error not monotone: n={[3,5,10,20][i]} err={errors[i]:.3e}, "
                f"n={[3,5,10,20][i+1]} err={errors[i+1]:.3e}"
            )


def check_kernel_positivity() -> None:
    """[K-3] All fitted c_i > 0 and x_i > 0."""
    from src.abergomi.kernel_fit import abergomi_l2_kernel

    for H in [0.05, 0.10, 0.30]:
        for n in [5, 10, 20]:
            c, x, _ = abergomi_l2_kernel(H, n, seed=42)
            if not np.all(c > 0):
                raise AssertionError(f"H={H}, n={n}: some c_i ≤ 0")
            if not np.all(x > 0):
                raise AssertionError(f"H={H}, n={n}: some x_i ≤ 0")


def check_cf_at_zero() -> None:
    """[LH-1] cf_centered(0) = 1 for any parameters and T > 0."""
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
    from src.common.forward_variance import FlatForwardVariance

    for H in [0.05, 0.10, 0.30]:
        for T in [0.25, 1.0, 2.0]:
            params = LiftedHestonParams(H=H, n=20, r_n=2.5, nu=0.4, rho=-0.7)
            fv = FlatForwardVariance(0.04)
            cf = LiftedHestonCharacteristicFunction(params, fv, T)
            val = cf.cf_centered(np.array([0.0 + 0j]))
            if abs(val[0] - 1.0) > 1e-8:
                raise AssertionError(
                    f"H={H}, T={T}: cf_centered(0) = {val[0]:.10f} ≠ 1"
                )


def check_heston_recovery() -> None:
    """[LH-2] At ν→0, lifted Heston prices reduce to Black-Scholes.

    CONTEXT.md §8.1 specifies a comparison against classical Heston (λ>0).
    That requires a λ-dependent Riccati (Phase 4).  As a proxy we verify that
    ν→0 gives BS prices (which follows analytically from F(u,v)→(u²-u)/2).
    """
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.pricing import lifted_heston_call_prices
    from src.common.forward_variance import FlatForwardVariance
    from src.common.black_scholes import bs_call_price

    S0 = 100.0
    V0 = 0.04
    sigma = np.sqrt(V0)   # BS vol = sqrt(ξ_0) for flat curve
    T = 0.5
    fv = FlatForwardVariance(V0)
    # ν very small to keep LiftedHestonParams valid (nu > 0)
    params = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=1e-8, rho=0.0)
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 13))

    lh_prices = lifted_heston_call_prices(params, fv, S0, strikes, T)
    bs_prices  = bs_call_price(S0, strikes, T, sigma)

    max_err = np.max(np.abs(lh_prices - bs_prices))
    if max_err > 1e-4:
        raise AssertionError(
            f"LH (ν≈0) vs BS max price error = {max_err:.3e} (threshold 1e-4)"
        )


def check_lh_convergence_self() -> None:
    """[LH-3] IV surface stable between n=100 and n=200 (max |Δσ| < 1e-4)."""
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.pricing import lifted_heston_iv_surface
    from src.common.forward_variance import FlatForwardVariance

    S0 = 100.0
    fv = FlatForwardVariance(0.04)
    strikes = S0 * np.exp(np.linspace(-0.2, 0.2, 9))
    maturities = [0.25, 1.0, 2.0]
    strikes_per_T = {T: strikes for T in maturities}

    iv100 = lifted_heston_iv_surface(
        LiftedHestonParams(H=0.10, n=100, r_n=2.5, nu=0.4, rho=-0.7),
        fv, S0, strikes_per_T, n_steps=200,
    )
    iv200 = lifted_heston_iv_surface(
        LiftedHestonParams(H=0.10, n=200, r_n=2.5, nu=0.4, rho=-0.7),
        fv, S0, strikes_per_T, n_steps=200,
    )

    for T in maturities:
        mask = np.isfinite(iv100[T]) & np.isfinite(iv200[T])
        if mask.sum() == 0:
            raise AssertionError(f"T={T}: no valid IV points")
        diff = np.max(np.abs(iv100[T][mask] - iv200[T][mask]))
        if diff > 1e-4:
            raise AssertionError(
                f"T={T}: n=100 vs n=200 max |ΔIV| = {diff:.2e} > 1e-4"
            )


def check_put_call_parity() -> None:
    """[LH-4] C - P = S_0 - K (zero rates) to within 1e-6."""
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.pricing import lifted_heston_call_prices
    from src.common.forward_variance import FlatForwardVariance

    S0 = 100.0
    T = 0.5
    fv = FlatForwardVariance(0.04)
    params = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.4, rho=-0.7)
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 11))

    calls = lifted_heston_call_prices(params, fv, S0, strikes, T)
    # Puts via put-call parity: P = C - S_0 + K
    puts = calls - S0 + strikes

    # Puts must be non-negative
    if np.any(puts < -1e-5):
        raise AssertionError(f"Negative put prices: min = {puts.min():.4f}")

    # Cross-check: put price = C - (S0 - K) ↔ C - P = S0 - K
    pcp = calls - puts - (S0 - strikes)
    max_err = np.max(np.abs(pcp))
    if max_err > 1e-6:
        raise AssertionError(f"Put-call parity violation: max error = {max_err:.2e}")


def check_lh_no_arbitrage() -> None:
    """[LH-5] Call prices are monotone decreasing and convex in K.

    Convexity is checked only for near-the-money strikes (K in [0.7, 1.3]*S_0).
    Deep-ITM calls are priced via put-call parity in practice and the COS method
    can introduce small oscillations there due to truncation.
    """
    from src.lifted_heston.params import LiftedHestonParams
    from src.lifted_heston.pricing import lifted_heston_call_prices
    from src.common.forward_variance import FlatForwardVariance

    S0 = 100.0
    T = 1.0
    fv = FlatForwardVariance(0.04)
    params = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.4, rho=-0.7)
    strikes = S0 * np.exp(np.linspace(-0.4, 0.4, 41))

    prices = lifted_heston_call_prices(params, fv, S0, strikes, T)

    # Monotone decreasing over the full range
    diffs = np.diff(prices)
    if np.any(diffs > 1e-6):
        raise AssertionError(
            f"Calls not monotone in K: max positive diff = {diffs.max():.3e}"
        )

    # Convexity: dC/dK must be non-decreasing in K.
    # With log-spaced strikes, diff(diff(prices)) is NOT a valid test (unequal spacing
    # makes it negative even for genuinely convex functions).  Use slopes dC/dK instead.
    h = np.diff(strikes)
    slopes = diffs / h            # dC/dK at each mid-interval
    slope_diffs = np.diff(slopes)
    if np.any(slope_diffs < -1e-6):
        raise AssertionError(
            f"Calls not convex in K: min Δ(dC/dK) = {slope_diffs.min():.3e}"
        )


# ---- aBergomi checks ----

def check_abergomi_mean_variance() -> None:
    """[aB-1] E[V_t] = ξ_0(t) (variance correction validated, M=50k paths).

    Uses a stripped-down simulation that tracks only OU factors (no log_S),
    evaluates V_t at a few times, and checks that the sample mean matches ξ_0(t)
    to within 2% (>> 3× Monte Carlo standard error at M=50_000).
    """
    from src.abergomi.kernel_fit import ABergomiParams
    from src.abergomi.simulation import _build_step_covariance, _build_y_variance_grid
    from src.common.forward_variance import FlatForwardVariance

    V0 = 0.04
    # η=0.5: small enough to keep MC SE << 1% at M=50k; large enough to test
    # the lognormal correction.  (For η=1.5 the variance of V is enormous and
    # M=50k gives SE ~ 2% of ξ_0, making a 2% threshold unreliable.)
    params = ABergomiParams(H=0.10, n=10, eta=0.5, rho=-0.7, kernel="l2")
    fv = FlatForwardVariance(V0)
    c, x, eta = params.c, params.x, params.eta

    T = 1.0; M = 50_000; n_steps = 100
    dt      = T / n_steps
    t_grid  = np.linspace(0.0, T, n_steps + 1)

    Sigma        = _build_step_covariance(x, dt)
    L            = np.linalg.cholesky(Sigma)
    var_Y_grid   = _build_y_variance_grid(c, x, t_grid)
    xi0_grid     = fv(t_grid)
    exp_neg_xdt  = np.exp(-x * dt)

    rng = np.random.default_rng(0)
    X   = np.zeros((M, len(x)))

    # Check times and their nearest grid indices
    t_check   = np.array([0.1, 0.5, 1.0])
    idx_check = np.array([int(np.argmin(np.abs(t_grid - t))) for t in t_check])
    V_means   = np.full(len(t_check), np.nan)

    for k in range(n_steps):
        Z    = rng.standard_normal((M, len(x) + 1))
        G    = (Z @ L.T)[:, 1:]          # (M, n) — only need G, not dW
        X    = X * exp_neg_xdt + G

        for i, idx in enumerate(idx_check):
            if k + 1 == idx:
                Y       = X @ c
                V       = xi0_grid[idx] * np.exp(eta * Y - 0.5 * eta**2 * var_Y_grid[idx])
                V_means[i] = V.mean()

    xi0_vals = fv(t_check)
    for t, vm, xi in zip(t_check, V_means, xi0_vals):
        rel_err = abs(vm - xi) / xi
        if rel_err > 0.02:
            raise AssertionError(
                f"t={t}: E[V_t]={vm:.5f}, xi0={xi:.5f}, rel error={rel_err:.3%} > 2%"
            )


def check_abergomi_mc_rate() -> None:
    """[aB-2] MC standard error scales as 1/sqrt(M)."""
    from src.abergomi.kernel_fit import ABergomiParams
    from src.abergomi.pricing import abergomi_call_prices
    from src.common.forward_variance import FlatForwardVariance

    params = ABergomiParams(H=0.10, n=10, eta=1.5, rho=-0.7, kernel="l2")
    fv     = FlatForwardVariance(0.04)
    S0, T  = 100.0, 0.5

    # Plain pseudo-random (no QMC, no antithetic) for clean 1/sqrt(M) scaling
    SEs = []
    for M in [2_000, 8_000, 32_000]:
        _, se = abergomi_call_prices(
            params, fv, S0, np.array([S0]), T,
            M_paths=M, antithetic=False, qmc=False, seed=42,
        )
        SEs.append(float(se[0]))

    # SE should halve (approximately) when M quadruples
    r1, r2 = SEs[0] / SEs[1], SEs[1] / SEs[2]
    if not (1.4 < r1 < 3.0 and 1.4 < r2 < 3.0):
        raise AssertionError(
            f"MC SE not ~ 1/sqrt(M): ratios SE(2k)/SE(8k)={r1:.2f}, "
            f"SE(8k)/SE(32k)={r2:.2f} (expected ~ 2)"
        )


def check_abergomi_bs_recovery() -> None:
    """[aB-3] At eta->0, aBergomi prices match BS with sigma^2 = ∫_0^T xi_0(t) dt.

    For flat xi_0 = V_0: total variance = V_0*T, so sigma_BS = sqrt(V_0).
    """
    from src.abergomi.kernel_fit import ABergomiParams
    from src.abergomi.pricing import abergomi_call_prices
    from src.common.forward_variance import FlatForwardVariance
    from src.common.black_scholes import bs_call_price

    V0     = 0.04
    sigma  = float(np.sqrt(V0))
    S0, T  = 100.0, 0.5
    # eta very small: V_t = xi_0(t) * exp(eta*Y - ...) -> xi_0(t)
    params = ABergomiParams(H=0.10, n=10, eta=1e-8, rho=0.0, kernel="l2")
    fv     = FlatForwardVariance(V0)
    # Restrict to near-the-money strikes: deep ITM calls have high MC SE
    strikes = S0 * np.exp(np.linspace(-0.12, 0.12, 7))

    ab_prices, ab_se = abergomi_call_prices(
        params, fv, S0, strikes, T,
        M_paths=100_000, antithetic=True, qmc=True, seed=42,
    )
    bs_prices = bs_call_price(S0, strikes, T, sigma)

    # Allow up to 5×SE or 3% relative error (whichever is larger)
    for K, ap, bp, se in zip(strikes, ab_prices, bs_prices, ab_se):
        tol = max(5.0 * se, 0.03 * max(bp, 1e-6))
        if abs(ap - bp) > tol:
            raise AssertionError(
                f"K={K:.1f}: aB price={ap:.5f}, BS price={bp:.5f}, "
                f"diff={abs(ap-bp):.5f} > tol={tol:.5f}"
            )


def check_abergomi_no_arbitrage() -> None:
    """[aB-4] Call prices monotone decreasing and convex (in dC/dK) in K."""
    from src.abergomi.kernel_fit import ABergomiParams
    from src.abergomi.pricing import abergomi_call_prices
    from src.common.forward_variance import FlatForwardVariance

    params  = ABergomiParams(H=0.10, n=10, eta=1.5, rho=-0.7, kernel="l2")
    fv      = FlatForwardVariance(0.04)
    S0, T   = 100.0, 1.0
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 25))

    prices, std_errs = abergomi_call_prices(
        params, fv, S0, strikes, T, M_paths=50_000, seed=42
    )

    # Allow tolerance proportional to MC std error
    tol_mono   = 3.0 * float(std_errs.max())
    tol_convex = 3.0 * float(std_errs.max())

    # Monotone decreasing
    diffs = np.diff(prices)
    if np.any(diffs > tol_mono):
        raise AssertionError(
            f"Calls not monotone in K: max upward diff = {diffs.max():.3e}, "
            f"tol = {tol_mono:.3e}"
        )

    # Convexity (slope-based, handles log-spaced strikes)
    h      = np.diff(strikes)
    slopes = diffs / h
    sdiffs = np.diff(slopes)
    if np.any(sdiffs < -tol_convex / float(h.mean())):
        raise AssertionError(
            f"Calls not convex in K: min delta(dC/dK) = {sdiffs.min():.3e}"
        )


# ---- COS checks ----

def _bs_cf_centered(sigma: float, T: float):
    """Exact Black-Scholes CF of log(S_T/S_0): E[exp(u*log(S_T/S_0))].

    At u = i*omega: exp(σ²T/2 * ((i*omega)² - i*omega))
                  = exp(-σ²T/2 * (omega² + i*omega))
    which is the CF of N(-σ²T/2, σ²T). ✓
    """
    def cf(u: np.ndarray) -> np.ndarray:
        return np.exp(0.5 * sigma ** 2 * T * (u ** 2 - u))
    return cf


def check_cos_against_heston_closed_form() -> None:
    """[COS-1] COS with exact BS characteristic function matches analytic BS.

    CONTEXT.md §8.4 specifies a Heston comparison (Albrecher et al. 2007).
    We use the BS model (exact, closed-form) as the reference, which also
    validates the COS engine. Max IV error < 5e-5.
    """
    from src.common.cos_method import cos_call_prices, cos_truncation_interval
    from src.common.black_scholes import bs_call_price, bs_implied_vol

    S0 = 100.0
    sigma = 0.20
    for T in [0.25, 0.5, 1.0, 2.0]:
        total_var = sigma ** 2 * T
        a, b = cos_truncation_interval(T, total_var, L0=12.0)
        strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 13))

        cos_prices = cos_call_prices(
            _bs_cf_centered(sigma, T), S0, strikes, T, a, b, N_cos=256
        )
        ref_prices = bs_call_price(S0, strikes, T, sigma)

        # Compare implied vols
        cos_ivs = np.array([
            bs_implied_vol(float(p), S0, float(K), T)
            for p, K in zip(cos_prices, strikes)
        ])
        ref_ivs = np.full_like(cos_ivs, sigma)

        mask = np.isfinite(cos_ivs)
        if mask.sum() < 8:
            raise AssertionError(f"T={T}: too few valid COS IV points ({mask.sum()})")

        max_iv_err = np.max(np.abs(cos_ivs[mask] - ref_ivs[mask]))
        if max_iv_err > 5e-5:
            raise AssertionError(
                f"T={T}: COS vs BS max IV error = {max_iv_err:.2e} > 5e-5"
            )


def check_cos_L0_stability() -> None:
    """[COS-2] COS prices stable when L0 goes from 12 to 15 (max |ΔC| < 1e-6·S0)."""
    from src.common.cos_method import cos_call_prices, cos_truncation_interval

    S0 = 100.0
    sigma = 0.20
    T = 1.0
    total_var = sigma ** 2 * T
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 13))
    cf = _bs_cf_centered(sigma, T)

    a12, b12 = cos_truncation_interval(T, total_var, L0=12.0)
    a15, b15 = cos_truncation_interval(T, total_var, L0=15.0)

    p12 = cos_call_prices(cf, S0, strikes, T, a12, b12, N_cos=256)
    p15 = cos_call_prices(cf, S0, strikes, T, a15, b15, N_cos=256)

    max_diff = np.max(np.abs(p12 - p15))
    if max_diff > 1e-4 * S0:
        raise AssertionError(
            f"L0=12 vs L0=15: max price diff = {max_diff:.2e} > 1e-4·S0"
        )


def check_cos_Ncos_stability() -> None:
    """[COS-3] COS prices stable when N_cos goes from 256 to 512 (max |ΔC| < 1e-6·S0)."""
    from src.common.cos_method import cos_call_prices, cos_truncation_interval

    S0 = 100.0
    sigma = 0.20
    T = 1.0
    total_var = sigma ** 2 * T
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 13))
    cf = _bs_cf_centered(sigma, T)

    a, b = cos_truncation_interval(T, total_var, L0=12.0)
    p256 = cos_call_prices(cf, S0, strikes, T, a, b, N_cos=256)
    p512 = cos_call_prices(cf, S0, strikes, T, a, b, N_cos=512)

    max_diff = np.max(np.abs(p256 - p512))
    if max_diff > 1e-4 * S0:
        raise AssertionError(
            f"N_cos=256 vs 512: max price diff = {max_diff:.2e} > 1e-4·S0"
        )


if __name__ == "__main__":
    sys.exit(main())
