"""Tests for Phase 4: rough Heston (fractional Riccati) and LH-L² variant."""
import numpy as np
import pytest

from src.rough_heston.fractional_riccati import solve_fractional_riccati
from src.rough_heston.pricing import rough_heston_cf_centered, rough_heston_iv_surface
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.common.forward_variance import FlatForwardVariance


# ── Fractional Riccati ──────────────────────────────────────────────────────

def test_fractional_riccati_initial_zero():
    """g(u, 0) = 0 by construction."""
    u = np.array([0.0+0j, 1j*1.0, 1j*5.0])
    t, g = solve_fractional_riccati(u, H=0.10, nu=0.4, rho=-0.7, T=0.5, n_steps=50)
    np.testing.assert_allclose(g[:, 0], 0.0, atol=1e-15)


def test_cf_at_zero_is_one():
    """cf_centered(0) = 1 for rough Heston, any H and T."""
    fv = FlatForwardVariance(0.04)
    for H, T in [(0.05, 0.5), (0.10, 1.0), (0.20, 2.0)]:
        cf = rough_heston_cf_centered(
            np.array([0.0+0j]), H=H, nu=0.4, rho=-0.7,
            forward_variance=fv, T=T, n_steps=50,
        )
        assert abs(cf[0] - 1.0) < 1e-8, f"H={H}, T={T}: cf(0) = {cf[0]}"


def test_cf_modulus_le_one():
    """|cf_centered(iω)| ≤ 1 for all real ω (characteristic function property)."""
    fv  = FlatForwardVariance(0.04)
    u   = 1j * np.array([0.0, 1.0, 5.0, 20.0])
    cf  = rough_heston_cf_centered(u, H=0.10, nu=0.4, rho=-0.7,
                                    forward_variance=fv, T=1.0, n_steps=100)
    assert np.all(np.abs(cf) <= 1.0 + 1e-6), f"max |cf| = {np.abs(cf).max():.4f}"


def test_rough_heston_converges_to_lh_large_n():
    """Rough Heston CF should be close to LH n=500 for the same parameters."""
    fv  = FlatForwardVariance(0.04)
    u   = 1j * np.array([1.0, 5.0, 10.0])

    cf_rh = rough_heston_cf_centered(u, H=0.10, nu=0.4, rho=-0.7,
                                      forward_variance=fv, T=0.5, n_steps=200)
    params500  = LiftedHestonParams(H=0.10, n=500, r_n=2.5, nu=0.4, rho=-0.7)
    cf_lh500   = LiftedHestonCharacteristicFunction(params500, fv, T=0.5, n_steps=200)
    cf_lh_vals = cf_lh500.cf_centered(u)

    max_diff = float(np.max(np.abs(cf_rh - cf_lh_vals)))
    assert max_diff < 0.05, (
        f"Rough Heston cf too far from LH n=500: max diff = {max_diff:.4f}"
    )


def test_rough_heston_iv_surface_shape():
    """rough_heston_iv_surface returns one entry per maturity with correct shapes."""
    fv   = FlatForwardVariance(0.04)
    S0   = 100.0
    K_arr = S0 * np.exp(np.linspace(-0.2, 0.2, 7))
    mats  = {0.5: K_arr, 1.0: K_arr}
    iv = rough_heston_iv_surface(
        H=0.10, nu=0.4, rho=-0.7, forward_variance=fv,
        S0=S0, strikes_per_T=mats, n_steps=50,
    )
    assert set(iv.keys()) == {0.5, 1.0}
    for T, ivs in iv.items():
        assert ivs.shape == (7,), f"T={T}: wrong shape {ivs.shape}"
        assert np.all(np.isfinite(ivs) | np.isnan(ivs))   # no infs


def test_rough_heston_iv_reasonable():
    """IV surface from rough Heston is close to LH n=500 for standard params."""
    fv  = FlatForwardVariance(0.04)
    S0  = 100.0
    K   = S0 * np.exp(np.linspace(-0.2, 0.2, 7))
    mats = {1.0: K}

    iv_rh = rough_heston_iv_surface(
        H=0.10, nu=0.4, rho=-0.7, forward_variance=fv,
        S0=S0, strikes_per_T=mats, n_steps=200,
    )
    params500 = LiftedHestonParams(H=0.10, n=500, r_n=2.5, nu=0.4, rho=-0.7)
    iv_lh500  = lifted_heston_iv_surface(params500, fv, S0, mats, n_steps=200)

    valid = np.isfinite(iv_rh[1.0]) & np.isfinite(iv_lh500[1.0])
    max_diff = float(np.max(np.abs(iv_rh[1.0][valid] - iv_lh500[1.0][valid])))
    assert max_diff < 0.01, f"Rough Heston vs LH500 max IV diff = {max_diff:.4f}"


# ── LH-L² kernel variant ────────────────────────────────────────────────────

def test_lh_l2_params_creation():
    """LiftedHestonParams with kernel='l2' creates valid (c, x)."""
    p = LiftedHestonParams(H=0.10, n=10, r_n=2.5, nu=0.4, rho=-0.7, kernel="l2")
    assert np.all(p.c > 0)
    assert np.all(p.x > 0)
    assert np.all(np.diff(p.x) > 0)
    assert len(p.c) == 10


def test_lh_l2_backward_compat():
    """Default kernel='geometric' is unchanged (backward compatible)."""
    p = LiftedHestonParams(H=0.10, n=10, r_n=2.5, nu=0.4, rho=-0.7)
    assert p.kernel == "geometric"


def test_lh_l2_gives_different_ivs_than_geo():
    """LH-L² and LH-geo produce different IV surfaces (different kernels)."""
    fv  = FlatForwardVariance(0.04)
    S0  = 100.0
    K   = S0 * np.exp(np.linspace(-0.2, 0.2, 7))
    mats = {1.0: K}

    p_geo = LiftedHestonParams(H=0.10, n=10, r_n=2.5, nu=0.4, rho=-0.7, kernel="geometric")
    p_l2  = LiftedHestonParams(H=0.10, n=10, r_n=2.5, nu=0.4, rho=-0.7, kernel="l2")

    iv_geo = lifted_heston_iv_surface(p_geo, fv, S0, mats)
    iv_l2  = lifted_heston_iv_surface(p_l2, fv, S0, mats)

    valid = np.isfinite(iv_geo[1.0]) & np.isfinite(iv_l2[1.0])
    assert valid.any(), "No valid IV points"
    diff = float(np.max(np.abs(iv_geo[1.0][valid] - iv_l2[1.0][valid])))
    assert diff > 1e-4, f"LH-geo and LH-L² IVs are identical (diff={diff:.2e})"


def test_lh_l2_iv_surface_no_arbitrage():
    """LH-L² call prices are monotone and convex in K."""
    fv  = FlatForwardVariance(0.04)
    S0  = 100.0
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 21))
    p   = LiftedHestonParams(H=0.10, n=10, r_n=2.5, nu=0.4, rho=-0.7, kernel="l2")

    from src.lifted_heston.pricing import lifted_heston_call_prices
    prices = lifted_heston_call_prices(p, fv, S0, strikes, T=1.0)

    # Monotone decreasing
    diffs = np.diff(prices)
    assert np.all(diffs <= 1e-6), f"Not monotone; max up-diff = {diffs.max():.3e}"

    # Convex via slopes (handles log-spaced strikes)
    h      = np.diff(strikes)
    slopes = diffs / h
    assert np.all(np.diff(slopes) >= -1e-6)
