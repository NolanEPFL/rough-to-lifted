"""Tests for src/common: Black-Scholes, forward variance, COS engine."""
import numpy as np
import pytest

from src.common.black_scholes import (
    bs_call_price, bs_put_price, bs_implied_vol, bs_vega
)
from src.common.forward_variance import FlatForwardVariance, PiecewiseConstantForwardVariance
from src.common.cos_method import cos_call_prices, cos_truncation_interval


# ---- Black-Scholes ----------------------------------------------------------

def test_bs_put_call_parity():
    """C - P = F - K * exp(-r T) (here r=0, so C - P = F - K)."""
    F, K, T, sigma = 100.0, 95.0, 0.5, 0.2
    C = bs_call_price(F, K, T, sigma)
    P = bs_put_price(F, K, T, sigma)
    assert abs(C - P - (F - K)) < 1e-10


def test_bs_atm_implied_vol_inversion():
    """Round-trip: price an ATM call, invert, recover sigma."""
    F, K, T, sigma_true = 100.0, 100.0, 0.25, 0.20
    price = float(bs_call_price(F, K, T, sigma_true))
    sigma_inv = bs_implied_vol(price, F, K, T)
    assert abs(sigma_inv - sigma_true) < 1e-8


def test_bs_implied_vol_out_of_bounds_returns_nan():
    """Returns nan when price is outside no-arbitrage bounds."""
    assert np.isnan(bs_implied_vol(-1.0, 100.0, 100.0, 1.0))
    assert np.isnan(bs_implied_vol(101.0, 100.0, 100.0, 1.0))


def test_bs_vega_positive():
    """Vega > 0 everywhere."""
    F = np.array([80.0, 100.0, 120.0])
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.25, 0.5, 1.0])
    sigma = np.array([0.15, 0.20, 0.25])
    v = bs_vega(F, K, T, sigma)
    assert np.all(v > 0)


def test_bs_call_monotone_in_sigma():
    """Call price is monotone increasing in sigma."""
    F, K, T = 100.0, 100.0, 1.0
    sigmas = np.linspace(0.05, 1.0, 20)
    prices = bs_call_price(F, K, T, sigmas)
    assert np.all(np.diff(prices) > 0)


def test_bs_put_intrinsic():
    """Put price >= max(K - F, 0)."""
    F, K, T, sigma = 100.0, 120.0, 1.0, 0.2
    P = float(bs_put_price(F, K, T, sigma))
    assert P >= K - F


# ---- Forward variance -------------------------------------------------------

def test_flat_forward_variance():
    fv = FlatForwardVariance(V0=0.04)
    assert fv(0.5) == 0.04
    assert fv.integrated(2.0) == pytest.approx(0.08)


def test_flat_forward_variance_array():
    fv = FlatForwardVariance(V0=0.04)
    t = np.array([0.1, 0.5, 1.0])
    np.testing.assert_allclose(fv(t), 0.04)


def test_piecewise_constant_call():
    knots = np.array([0.5, 1.0, 2.0])
    levels = np.array([0.04, 0.05, 0.03])
    fv = PiecewiseConstantForwardVariance(knots, levels)
    assert fv(0.0) == pytest.approx(0.04)
    assert fv(0.3) == pytest.approx(0.04)
    assert fv(0.5) == pytest.approx(0.05)   # cadlag: right-continuous
    assert fv(0.8) == pytest.approx(0.05)
    assert fv(1.0) == pytest.approx(0.03)
    assert fv(1.5) == pytest.approx(0.03)
    assert fv(3.0) == pytest.approx(0.03)   # extrapolation


def test_piecewise_constant_integrated():
    knots = np.array([0.5, 1.0, 2.0])
    levels = np.array([0.04, 0.05, 0.03])
    fv = PiecewiseConstantForwardVariance(knots, levels)
    assert fv.integrated(0.25) == pytest.approx(0.04 * 0.25)
    assert fv.integrated(0.5) == pytest.approx(0.04 * 0.5)
    assert fv.integrated(0.75) == pytest.approx(0.04 * 0.5 + 0.05 * 0.25)
    assert fv.integrated(1.5) == pytest.approx(0.04 * 0.5 + 0.05 * 0.5 + 0.03 * 0.5)
    # Extrapolation
    assert fv.integrated(2.5) == pytest.approx(
        0.04 * 0.5 + 0.05 * 0.5 + 0.03 * 1.0 + 0.03 * 0.5
    )


# ---- COS pricing ------------------------------------------------------------

def _bs_cf(sigma, T):
    """Exact BS CF of log(S_T/S_0): E[exp(u * log(S_T/S_0))]."""
    def cf(u):
        return np.exp(0.5 * sigma ** 2 * T * (u ** 2 - u))
    return cf


def test_cos_against_bs():
    """COS pricing of a Black-Scholes characteristic function should recover BS."""
    S0, sigma, T = 100.0, 0.20, 1.0
    total_var = sigma ** 2 * T
    a, b = cos_truncation_interval(T, total_var, L0=12.0)
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 13))

    cos_p = cos_call_prices(_bs_cf(sigma, T), S0, strikes, T, a, b, N_cos=256)
    ref_p = bs_call_price(S0, strikes, T, sigma)

    np.testing.assert_allclose(cos_p, ref_p, atol=1e-5, rtol=0)


def test_cos_truncation_stability():
    """Prices stable when L0 is increased from 12 to 15."""
    S0, sigma, T = 100.0, 0.20, 1.0
    total_var = sigma ** 2 * T
    cf = _bs_cf(sigma, T)
    strikes = S0 * np.exp(np.linspace(-0.3, 0.3, 11))

    a12, b12 = cos_truncation_interval(T, total_var, L0=12.0)
    a15, b15 = cos_truncation_interval(T, total_var, L0=15.0)
    p12 = cos_call_prices(cf, S0, strikes, T, a12, b12)
    p15 = cos_call_prices(cf, S0, strikes, T, a15, b15)
    np.testing.assert_allclose(p12, p15, atol=1e-5 * S0)
