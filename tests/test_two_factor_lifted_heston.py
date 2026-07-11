"""Tests for the two-factor lifted Heston model."""

from __future__ import annotations

import numpy as np
import pytest

from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.cir_riccati import (
    cir_riccati_F, solve_cir_riccati, cir_cf_centered,
)
from src.two_factor_lifted_heston.characteristic_function import (
    TwoFactorLiftedHestonCF, block1_forward_variance,
)
from src.two_factor_lifted_heston.pricing import (
    two_factor_lh_call_prices, two_factor_lh_iv_surface,
)
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_call_prices
from src.common.forward_variance import FlatForwardVariance


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_params():
    return TwoFactorLiftedHestonParams(
        H1=0.10, n1=20, r_n1=2.5, nu1=0.3, rho1=-0.7,
        lam2=1.0, theta2=0.02, nu2=0.2, rho2=-0.4, V2_0=0.02,
    )


@pytest.fixture
def flat_fv():
    return FlatForwardVariance(0.04)


# ── Params ────────────────────────────────────────────────────────────────────

def test_params_valid(base_params):
    assert base_params.H1 == 0.10
    assert len(base_params.c) == 20
    assert len(base_params.x) == 20
    assert np.all(base_params.c > 0)
    assert np.all(base_params.x > 0)


def test_params_invalid_H():
    with pytest.raises(ValueError):
        TwoFactorLiftedHestonParams(
            H1=0.6, n1=20, r_n1=2.5, nu1=0.3, rho1=-0.7,
            lam2=1.0, theta2=0.02, nu2=0.2, rho2=-0.4, V2_0=0.02,
        )


def test_block2_mean_at_zero(base_params):
    assert abs(base_params.block2_mean(0.0) - base_params.V2_0) < 1e-12


def test_block2_mean_at_infinity(base_params):
    # Should converge to theta2 as t -> inf
    assert abs(base_params.block2_mean(1000.0) - base_params.theta2) < 1e-6


# ── CIR Riccati ───────────────────────────────────────────────────────────────

def test_cir_riccati_zero_initial():
    u = np.array([1.0 + 1.0j, 2.0 + 0.5j])
    t_grid, psi2 = solve_cir_riccati(u, nu2=0.3, rho2=-0.4, lam2=1.0, T=1.0)
    assert psi2.shape == (2, 201)
    assert np.allclose(psi2[:, 0], 0.0)


def test_cir_cf_normalisation_u0():
    # Phi^(2)(u=0) = exp(0) = 1
    phi = cir_cf_centered(
        np.array([0.0 + 0.0j]),
        nu2=0.3, rho2=-0.4, lam2=1.0, theta2=0.02, V2_0=0.02, T=1.0,
    )
    assert abs(complex(phi[0]) - 1.0) < 1e-10


def test_cir_cf_normalisation_u1():
    # Phi^(2)(u=1) = 1 (martingale property of the CIR block)
    phi = cir_cf_centered(
        np.array([1.0 + 0.0j]),
        nu2=0.3, rho2=-0.4, lam2=1.0, theta2=0.02, V2_0=0.02, T=1.0,
    )
    assert abs(complex(phi[0]) - 1.0) < 1e-8


def test_cir_cf_nu2_zero():
    # nu2=0 => block 2 is deterministic, CF reduces to exp(V2_0*psi(T) + ...)
    # Just check it runs and returns finite values
    phi = cir_cf_centered(
        np.array([1.0j, 2.0j]),
        nu2=0.0, rho2=0.0, lam2=1.0, theta2=0.02, V2_0=0.02, T=1.0,
    )
    assert np.all(np.isfinite(np.abs(phi)))


# ── block1_forward_variance ───────────────────────────────────────────────────

def test_block1_fv_positive(base_params, flat_fv):
    fv1 = block1_forward_variance(flat_fv, base_params)
    t_vals = np.linspace(0.01, 2.0, 50)
    xi1 = fv1(t_vals)
    assert np.all(xi1 > 0), "block1 forward variance must be positive"


def test_block1_fv_vanishing_block2():
    # If V2_0 = theta2 = 0 and nu2 = 0, block2_mean = 0, so xi1 = xi_market
    params = TwoFactorLiftedHestonParams(
        H1=0.10, n1=20, r_n1=2.5, nu1=0.3, rho1=-0.7,
        lam2=1.0, theta2=0.0, nu2=0.0, rho2=0.0, V2_0=0.0,
    )
    fv = FlatForwardVariance(0.04)
    fv1 = block1_forward_variance(fv, params)
    t = np.array([0.1, 0.5, 1.0])
    assert np.allclose(fv1(t), 0.04)


# ── Two-factor CF normalisation ───────────────────────────────────────────────

def test_cf_normalisation_u0(base_params, flat_fv):
    cf = TwoFactorLiftedHestonCF(base_params, flat_fv, T=1.0)
    phi = cf.cf_centered(np.array([0.0 + 0.0j]))
    assert abs(complex(phi[0]) - 1.0) < 1e-8


def test_cf_normalisation_u1(base_params, flat_fv):
    cf = TwoFactorLiftedHestonCF(base_params, flat_fv, T=1.0)
    phi = cf.cf_centered(np.array([1.0 + 0.0j]))
    assert abs(complex(phi[0]) - 1.0) < 1e-8


# ── Recover single-block lifted Heston ───────────────────────────────────────

def test_recover_single_lh(flat_fv):
    """V2_0 = theta2 = nu2 = 0 => two-factor CF = single-block LH CF."""
    params_tf = TwoFactorLiftedHestonParams(
        H1=0.10, n1=20, r_n1=2.5, nu1=0.3, rho1=-0.7,
        lam2=1.0, theta2=0.0, nu2=0.0, rho2=0.0, V2_0=0.0,
    )
    params_lh = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.3, rho=-0.7)

    S0 = 100.0
    strikes = S0 * np.exp(np.linspace(-0.3, 0.2, 7))

    for T in [0.5, 1.0]:
        p_tf = two_factor_lh_call_prices(params_tf, flat_fv, S0, strikes, T)
        p_lh = lifted_heston_call_prices(params_lh, flat_fv, S0, strikes, T)
        assert np.allclose(p_tf, p_lh, atol=1e-10), (
            f"T={T}: max diff = {np.max(np.abs(p_tf - p_lh)):.2e}"
        )


# ── Pricing sanity ────────────────────────────────────────────────────────────

def test_call_prices_positive(base_params, flat_fv):
    strikes = np.array([90.0, 100.0, 110.0])
    prices = two_factor_lh_call_prices(base_params, flat_fv, 100.0, strikes, 1.0)
    assert np.all(prices >= 0)


def test_call_prices_decreasing_in_strike(base_params, flat_fv):
    strikes = np.array([85.0, 95.0, 105.0, 115.0])
    prices = two_factor_lh_call_prices(base_params, flat_fv, 100.0, strikes, 1.0)
    assert np.all(np.diff(prices) < 0)


def test_iv_surface_shape(base_params, flat_fv):
    strikes_per_T = {
        0.5: np.array([95.0, 100.0, 105.0]),
        1.0: np.array([90.0, 100.0, 110.0]),
    }
    iv = two_factor_lh_iv_surface(base_params, flat_fv, 100.0, strikes_per_T)
    assert set(iv.keys()) == {0.5, 1.0}
    assert len(iv[0.5]) == 3
    assert np.all(np.isfinite(iv[0.5]))
    assert np.all(iv[0.5] > 0)


def test_iv_reasonable_range(base_params, flat_fv):
    strikes = np.array([95.0, 100.0, 105.0])
    iv = two_factor_lh_iv_surface(
        base_params, flat_fv, 100.0, {1.0: strikes}
    )[1.0]
    assert np.all(iv > 0.05)
    assert np.all(iv < 1.0)
