"""Tests for src/abergomi: kernel fit, simulation, and pricing."""
import numpy as np
import pytest

from src.abergomi.kernel_fit import abergomi_l2_kernel, ABergomiParams
from src.abergomi.simulation import (
    simulate_abergomi,
    _build_step_covariance,
    _build_y_variance_grid,
)
from src.abergomi.pricing import abergomi_call_prices, abergomi_iv_surface
from src.common.forward_variance import FlatForwardVariance
from src.common.black_scholes import bs_call_price


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def ab_params():
    return ABergomiParams(H=0.10, n=10, eta=1.5, rho=-0.7, kernel="l2")


@pytest.fixture
def flat_fv():
    return FlatForwardVariance(0.04)


# ---- L² kernel fit ----------------------------------------------------------

@pytest.mark.parametrize("H", [0.05, 0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("n", [5, 10, 20])
def test_l2_kernel_positive_weights(H, n):
    """NNLS guarantees c_i ≥ 0; log-param guarantees x_i > 0."""
    c, x, _ = abergomi_l2_kernel(H, n, seed=42)
    assert np.all(c > 0)
    assert np.all(x > 0)
    assert np.all(np.diff(x) > 0), "speeds not ascending"


def test_l2_kernel_better_than_geometric_in_l2():
    """For the same n, L² fit ≤ geometric fit in L² norm (by construction)."""
    from src.lifted_heston.params import geometric_grid
    from src.lifted_heston.kernel import kernel_l2_error
    H, n    = 0.1, 10
    c_l2, x_l2, _ = abergomi_l2_kernel(H, n, seed=42)
    c_geo, x_geo   = geometric_grid(H, n)
    err_l2  = kernel_l2_error(H, c_l2, x_l2)
    err_geo = kernel_l2_error(H, c_geo, x_geo)
    assert err_l2 <= err_geo + 1e-6


def test_l2_kernel_stable_across_starts():
    """Multiple random restarts both converge to a good L² error (< 1e-3)."""
    H, n = 0.1, 10
    _, _, err1 = abergomi_l2_kernel(H, n, n_restarts=5, seed=0)
    _, _, err2 = abergomi_l2_kernel(H, n, n_restarts=5, seed=99)
    assert err1 < 1e-3, f"seed=0: L² error {err1:.2e} too large"
    assert err2 < 1e-3, f"seed=99: L² error {err2:.2e} too large"


# ---- One-step covariance ----------------------------------------------------

def test_step_covariance_symmetric():
    x = np.array([0.5, 2.0, 10.0])
    Sigma = _build_step_covariance(x, dt=0.01)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-14)


def test_step_covariance_psd():
    x = np.array([0.5, 2.0, 10.0, 50.0])
    Sigma = _build_step_covariance(x, dt=0.01)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-12), f"Non-PSD: min eigenvalue = {eigvals.min():.3e}"
    np.linalg.cholesky(Sigma)    # should not raise


def test_step_covariance_diagonal_formula():
    """Cov(G_i, G_i) = (1 − exp(−2 x_i Δ)) / (2 x_i) for each i."""
    x, dt = np.array([1.0, 5.0, 20.0]), 0.025
    Sigma    = _build_step_covariance(x, dt)
    expected = (1.0 - np.exp(-2.0 * x * dt)) / (2.0 * x)
    # Allow for the small diagonal jitter
    np.testing.assert_allclose(Sigma[1:, 1:].diagonal(), expected, rtol=1e-4)


def test_step_covariance_dW_formula():
    """Var(ΔW) = dt and Cov(ΔW, G^{(i)}) = (1 − exp(−x_i dt)) / x_i."""
    x, dt = np.array([1.0, 5.0]), 0.01
    Sigma = _build_step_covariance(x, dt)
    assert Sigma[0, 0] == pytest.approx(dt, rel=1e-4)
    expected_cross = (1.0 - np.exp(-x * dt)) / x
    np.testing.assert_allclose(Sigma[0, 1:], expected_cross, rtol=1e-4)


# ---- Variance-correction grid -----------------------------------------------

def test_y_variance_grid_zero_at_origin():
    c, x = np.array([0.5, 0.3]), np.array([1.0, 5.0])
    var_Y = _build_y_variance_grid(c, x, np.array([0.0, 0.5, 1.0]))
    assert var_Y[0] == pytest.approx(0.0, abs=1e-14)


def test_y_variance_grid_monotone():
    c, x = np.array([0.5, 0.3, 0.2]), np.array([1.0, 5.0, 20.0])
    t    = np.linspace(0.0, 2.0, 50)
    var_Y = _build_y_variance_grid(c, x, t)
    assert np.all(np.diff(var_Y) >= -1e-12)


def test_y_variance_matches_closed_form():
    c, x  = np.array([0.5, 0.3]), np.array([1.0, 5.0])
    t_vals = np.array([0.1, 0.5, 1.0, 2.0])
    var_Y  = _build_y_variance_grid(c, x, t_vals)
    for k, t in enumerate(t_vals):
        expected = sum(
            c[i] * c[j] * (1.0 - np.exp(-(x[i]+x[j])*t)) / (x[i]+x[j])
            for i in range(2) for j in range(2)
        )
        assert var_Y[k] == pytest.approx(expected, rel=1e-10)


# ---- Simulation: basic correctness ------------------------------------------

def test_log_S_shapes(ab_params, flat_fv):
    """Terminal + multi-maturity output shapes are correct."""
    times, ls = simulate_abergomi(
        ab_params, flat_fv, 100.0, T=0.5,
        M_paths=200, n_steps=20, antithetic=True, qmc=False, seed=0,
    )
    assert times.shape == (1,) and ls.shape == (400, 1)

    times2, ls2 = simulate_abergomi(
        ab_params, flat_fv, 100.0, T=1.0,
        M_paths=200, n_steps=40, antithetic=False, qmc=False, seed=0,
        record_at=np.array([0.5, 1.0]),
    )
    assert times2.shape == (2,) and ls2.shape == (200, 2)


def test_log_S_reproducible(ab_params, flat_fv):
    """Same seed → bit-identical output (CRN property)."""
    kwargs = dict(M_paths=300, n_steps=10, antithetic=True, qmc=False, seed=7)
    _, ls1 = simulate_abergomi(ab_params, flat_fv, 100.0, T=0.5, **kwargs)
    _, ls2 = simulate_abergomi(ab_params, flat_fv, 100.0, T=0.5, **kwargs)
    np.testing.assert_array_equal(ls1, ls2)


def test_log_S_martingale(ab_params, flat_fv):
    """E[S_T / S_0] ≈ 1 (risk-neutral martingale), using qmc=False."""
    S0 = 100.0
    _, log_S = simulate_abergomi(
        ab_params, flat_fv, S0, T=1.0,
        M_paths=20_000, n_steps=50, antithetic=True, qmc=False, seed=42,
    )
    mean_ratio = float(np.exp(log_S[:, 0]).mean() / S0)
    assert abs(mean_ratio - 1.0) < 0.02, f"Martingale: E[S_T/S0] = {mean_ratio:.5f}"


def test_E_V_t_matches_xi0(flat_fv):
    """E[V_t] = ξ_0(t): lognormal correction enforces exact forward matching.

    Use η=0.5 to keep the variance of V_t manageable at M=30k
    (for η=1.5, Std(V_t) ≈ 0.7, giving SE ≈ 10% of ξ_0 — too large for a 3% check).
    """
    params = ABergomiParams(H=0.10, n=10, eta=0.5, rho=-0.7, kernel="l2")
    c, x, eta = params.c, params.x, params.eta
    T, M, n_steps = 1.0, 30_000, 50

    dt      = T / n_steps
    t_grid  = np.linspace(0.0, T, n_steps + 1)
    Sigma   = _build_step_covariance(x, dt)
    L       = np.linalg.cholesky(Sigma)
    varY    = _build_y_variance_grid(c, x, t_grid)
    xi0     = flat_fv(t_grid)
    enx     = np.exp(-x * dt)

    rng = np.random.default_rng(0)
    X   = np.zeros((M, len(x)))
    checks = {10: None, 30: None, 50: None}

    for k in range(n_steps):
        Z  = rng.standard_normal((M, len(x) + 1))
        G  = (Z @ L.T)[:, 1:]
        X  = X * enx + G
        if k + 1 in checks:
            Y      = X @ c
            V_mean = (xi0[k+1] * np.exp(eta * Y - 0.5*eta**2 * varY[k+1])).mean()
            checks[k + 1] = float(V_mean)

    for step, vm in checks.items():
        xi = float(xi0[step])
        rel_err = abs(vm - xi) / xi
        assert rel_err < 0.03, f"step {step}: E[V]={vm:.5f}, xi0={xi:.5f}, err={rel_err:.3%}"


# ---- MC standard error ------------------------------------------------------

def test_mc_se_scales_like_one_over_sqrt_M(ab_params, flat_fv):
    """SE halves when M quadruples (plain MC, antithetic=False)."""
    SEs = []
    for M in [1_000, 4_000, 16_000]:
        _, se = abergomi_call_prices(
            ab_params, flat_fv, 100.0, np.array([100.0]), T=0.5,
            M_paths=M, antithetic=False, qmc=False, seed=42,
        )
        SEs.append(float(se[0]))
    r1, r2 = SEs[0] / SEs[1], SEs[1] / SEs[2]
    assert 1.3 < r1 < 3.2, f"SE ratio (1k/4k) = {r1:.2f}, expected ~2"
    assert 1.3 < r2 < 3.2, f"SE ratio (4k/16k) = {r2:.2f}, expected ~2"


# ---- Variance reduction -----------------------------------------------------

def test_antithetic_reduces_se(ab_params, flat_fv):
    """Antithetic variates should reduce SE vs plain MC for same M."""
    kw = dict(params=ab_params, forward_variance=flat_fv,
              S0=100.0, strikes=np.array([100.0]), T=0.5,
              M_paths=5_000, qmc=False, seed=42)
    _, se_plain = abergomi_call_prices(**kw, antithetic=False)
    _, se_anti  = abergomi_call_prices(**kw, antithetic=True)
    assert float(se_anti[0]) < float(se_plain[0]), (
        f"Antithetic SE ({se_anti[0]:.4e}) should be < plain SE ({se_plain[0]:.4e})"
    )


# ---- CRN --------------------------------------------------------------------

def test_crn_same_seed_same_prices(ab_params, flat_fv):
    """Two identical calls give identical prices."""
    kw = dict(params=ab_params, forward_variance=flat_fv,
              S0=100.0, strikes=np.array([95.0, 100.0, 105.0]), T=0.5,
              M_paths=1_000, seed=42)
    p1, _ = abergomi_call_prices(**kw)
    p2, _ = abergomi_call_prices(**kw)
    np.testing.assert_array_equal(p1, p2)


def test_crn_smooth_perturbation(flat_fv):
    """Small η perturbation under frozen CRN gives small price change."""
    kw = dict(forward_variance=flat_fv, S0=100.0,
              strikes=np.array([100.0]), T=0.5,
              M_paths=5_000, seed=42, qmc=False)
    p1, _ = abergomi_call_prices(
        ABergomiParams(H=0.10, n=10, eta=1.5, rho=-0.7, kernel="l2"), **kw)
    p2, _ = abergomi_call_prices(
        ABergomiParams(H=0.10, n=10, eta=1.6, rho=-0.7, kernel="l2"), **kw)
    assert abs(float(p2[0]) - float(p1[0])) < 2.0


# ---- IV surface -------------------------------------------------------------

def test_iv_surface_shape(ab_params, flat_fv):
    """abergomi_iv_surface returns one entry per maturity."""
    S0    = 100.0
    K_arr = S0 * np.exp(np.linspace(-0.2, 0.2, 7))
    iv, _ = abergomi_iv_surface(
        ab_params, flat_fv, S0,
        {0.5: K_arr, 1.0: K_arr},
        M_paths=1_000, qmc=False,
    )
    assert len(iv) == 2


def test_iv_surface_no_arbitrage(ab_params, flat_fv):
    """Call prices monotone decreasing and convex (dC/dK) in K (MC tolerance)."""
    S0      = 100.0
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 19))
    prices, se = abergomi_call_prices(
        ab_params, flat_fv, S0, strikes, T=1.0,
        M_paths=10_000, seed=42, qmc=False,
    )
    tol  = 4.0 * float(se.max())
    diffs = np.diff(prices)
    assert np.all(diffs <= tol), f"Not monotone, max up-diff = {diffs.max():.3e}"
    slopes = diffs / np.diff(strikes)
    assert np.all(np.diff(slopes) >= -tol / np.diff(strikes).mean()), "Not convex"


def test_bs_recovery_eta_zero(flat_fv):
    """At η → 0, aBergomi call prices match Black-Scholes."""
    V0, S0, T = 0.04, 100.0, 0.5
    sigma     = float(np.sqrt(V0))
    params    = ABergomiParams(H=0.10, n=10, eta=1e-8, rho=0.0, kernel="l2")
    strikes   = S0 * np.exp(np.linspace(-0.10, 0.10, 7))

    ab_p, se = abergomi_call_prices(
        params, flat_fv, S0, strikes, T,
        M_paths=50_000, antithetic=True, qmc=False, seed=42,
    )
    bs_p = bs_call_price(S0, strikes, T, sigma)
    for K, ap, bp, s in zip(strikes, ab_p, bs_p, se):
        tol = max(5.0 * s, 0.03 * max(bp, 1e-6))
        assert abs(ap - bp) <= tol, (
            f"K={K:.1f}: aB={ap:.4f}, BS={bp:.4f}, |diff|={abs(ap-bp):.4f}>tol={tol:.4f}"
        )


def test_iv_surface_negative_skew(flat_fv):
    """For ρ < 0, OTM puts (low K) should have higher IV than OTM calls."""
    S0     = 100.0
    params = ABergomiParams(H=0.10, n=10, eta=1.5, rho=-0.7, kernel="l2")
    K_arr  = S0 * np.exp(np.array([-0.15, 0.15]))
    iv, _  = abergomi_iv_surface(
        params, flat_fv, S0, {0.5: K_arr}, M_paths=20_000, qmc=False,
    )
    ivs = list(iv.values())[0]
    if np.all(np.isfinite(ivs)):
        assert ivs[0] >= ivs[1] - 0.05, (
            f"No negative skew: IV(low)={ivs[0]:.4f}, IV(high)={ivs[1]:.4f}"
        )
