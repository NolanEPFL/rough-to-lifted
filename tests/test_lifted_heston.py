"""Tests for src/lifted_heston."""
import numpy as np
import pytest

from src.lifted_heston.params import geometric_grid, LiftedHestonParams
from src.lifted_heston.kernel import K_H, K_n, kernel_l2_error
from src.lifted_heston.riccati import solve_riccati, riccati_F
from src.lifted_heston.characteristic_function import LiftedHestonCharacteristicFunction
from src.lifted_heston.pricing import lifted_heston_iv_surface, lifted_heston_call_prices
from src.common.forward_variance import FlatForwardVariance
from src.common.black_scholes import bs_call_price


# ---- Parameter grid ---------------------------------------------------------

@pytest.mark.parametrize("H", [0.05, 0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("n", [4, 10, 20])
def test_geometric_grid_positivity(H, n):
    """All weights and speeds strictly positive, speeds sorted."""
    c, x = geometric_grid(H, n)
    assert np.all(c > 0), "Some c_i ≤ 0"
    assert np.all(x > 0), "Some x_i ≤ 0"
    assert np.all(np.diff(x) > 0), "x_i not strictly ascending"
    assert len(c) == n
    assert len(x) == n


def test_geometric_grid_n_even_required():
    """Odd n should raise ValueError."""
    with pytest.raises(ValueError):
        geometric_grid(H=0.1, n=5)


def test_geometric_grid_H_bounds():
    """H outside (0, 0.5) should raise ValueError."""
    with pytest.raises(ValueError):
        geometric_grid(H=0.0, n=4)
    with pytest.raises(ValueError):
        geometric_grid(H=0.5, n=4)
    with pytest.raises(ValueError):
        geometric_grid(H=-0.1, n=4)


# ---- Kernel -----------------------------------------------------------------

def test_K_H_known_values():
    """K_H(t) = t^{H-1/2} / Γ(H+1/2) — check at a few points."""
    from scipy.special import gamma as Gamma

    for H in [0.05, 0.1, 0.2, 0.3]:
        t_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
        expected = t_vals ** (H - 0.5) / Gamma(H + 0.5)
        got = K_H(t_vals, H)
        np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_K_n_at_zero_equals_sum_c():
    """K_n(0) = Σ c_i (the sum of all weights)."""
    c = np.array([0.5, 0.3, 0.2])
    x = np.array([0.1, 1.0, 10.0])
    assert K_n(0.0, c, x) == pytest.approx(c.sum())


def test_K_n_vectorised():
    """K_n returns shape (T,) for t shape (T,)."""
    c = np.array([0.5, 0.3])
    x = np.array([1.0, 5.0])
    t = np.linspace(0.01, 2.0, 20)
    result = K_n(t, c, x)
    assert result.shape == (20,)
    assert np.all(result > 0)


def test_K_n_recovers_K_H_geometric_for_large_n():
    """For n=20 geometric, K_n approximates K_H; error < 0.1 and decreasing in n."""
    H = 0.1
    err20 = kernel_l2_error(H, *geometric_grid(H, n=20))
    err50 = kernel_l2_error(H, *geometric_grid(H, n=50))
    assert err20 < 0.1, f"n=20 L² error too large: {err20:.4f}"
    assert err50 < err20, "Error should decrease from n=20 to n=50"


def test_kernel_l2_error_decreases_in_n():
    """L² error from geometric grid decreases as n increases."""
    H = 0.1
    errors = [kernel_l2_error(H, *geometric_grid(H, n)) for n in [4, 10, 20]]
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


# ---- Riccati ----------------------------------------------------------------

def test_riccati_F_at_zero():
    """F(0, v) = 0.5 ν² v²."""
    nu, rho = 0.4, -0.7
    v = np.array([0.0, 1.0, 2.0])
    u = np.zeros(3, dtype=complex)
    result = riccati_F(u, v, nu, rho)
    expected = 0.5 * nu ** 2 * v ** 2
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_riccati_zero_at_t_zero():
    """ψ^{n,i}(0) = 0 by construction."""
    n, N_u = 4, 8
    c, x = geometric_grid(0.1, n)
    u_grid = 1j * np.linspace(0, 5, N_u)
    t_grid, psi = solve_riccati(u_grid, c, x, nu=0.4, rho=-0.7, T=1.0)
    np.testing.assert_allclose(psi[:, :, 0], 0.0, atol=1e-15)


def test_riccati_n_steps_convergence():
    """Riccati ψ values converge as n_steps increases (first-order scheme)."""
    n = 10
    c, x = geometric_grid(0.1, n)
    u_grid = 1j * np.array([0.0, 1.0, 5.0])

    _, psi400  = solve_riccati(u_grid, c, x, 0.4, -0.7, T=1.0, n_steps=400)
    _, psi800  = solve_riccati(u_grid, c, x, 0.4, -0.7, T=1.0, n_steps=800)
    _, psi1600 = solve_riccati(u_grid, c, x, 0.4, -0.7, T=1.0, n_steps=1600)

    diff1 = np.max(np.abs(psi400[:, :, -1] - psi800[:, :, -1]))
    diff2 = np.max(np.abs(psi800[:, :, -1] - psi1600[:, :, -1]))
    # First-order convergence: diff2 should be roughly half of diff1
    assert diff2 < diff1, "Riccati not converging with increasing n_steps"
    assert diff2 < 5e-3, f"Residual at n_steps=1600 too large: {diff2:.3e}"


# ---- Characteristic function ------------------------------------------------

@pytest.fixture
def default_lh_params():
    return LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=0.4, rho=-0.7)


def test_cf_at_zero_is_one(default_lh_params):
    """cf_centered(0) = 1 for various T."""
    fv = FlatForwardVariance(0.04)
    for T in [0.25, 1.0, 2.0]:
        cf = LiftedHestonCharacteristicFunction(default_lh_params, fv, T)
        val = cf.cf_centered(np.array([0.0 + 0j]))
        assert abs(val[0] - 1.0) < 1e-8, f"T={T}: cf(0) = {val[0]}"


def test_cf_bs_recovery_at_nu_zero():
    """At ν→0, cf_centered should match Black-Scholes CF."""
    sigma, T, V0 = 0.2, 1.0, 0.04
    fv = FlatForwardVariance(V0)
    params = LiftedHestonParams(H=0.10, n=20, r_n=2.5, nu=1e-8, rho=0.0)
    cf = LiftedHestonCharacteristicFunction(params, fv, T)

    omega = np.array([1.0, 5.0, 10.0])
    u = 1j * omega
    got = cf.cf_centered(u)
    expected = np.exp(0.5 * sigma ** 2 * T * (u ** 2 - u))
    np.testing.assert_allclose(np.real(got), np.real(expected), atol=1e-4)


# ---- IV surface -------------------------------------------------------------

@pytest.fixture
def default_spx_strikes():
    return np.linspace(-0.4, 0.2, 21)


@pytest.fixture
def default_maturities():
    return np.array([0.083, 0.25, 0.5, 1.0, 2.0])


def test_iv_surface_no_arbitrage(default_lh_params):
    """Call prices are monotone decreasing and convex (in dC/dK) in K."""
    S0 = 100.0
    fv = FlatForwardVariance(0.04)
    T = 1.0
    strikes = S0 * np.exp(np.linspace(-0.4, 0.4, 41))

    prices = lifted_heston_call_prices(default_lh_params, fv, S0, strikes, T)

    # Monotone decreasing
    assert np.all(np.diff(prices) <= 1e-6), "Calls not monotone in K"

    # Convexity via slope (handles log-spaced strikes)
    h = np.diff(strikes)
    slopes = np.diff(prices) / h
    assert np.all(np.diff(slopes) >= -1e-6), "Calls not convex in K"


def test_iv_surface_n_convergence():
    """IV stable from n=100 to n=200 (within 1e-4 absolute)."""
    S0 = 100.0
    fv = FlatForwardVariance(0.04)
    strikes = S0 * np.exp(np.linspace(-0.2, 0.2, 9))
    mats = {T: strikes for T in [0.5, 1.0, 2.0]}

    iv100 = lifted_heston_iv_surface(
        LiftedHestonParams(H=0.10, n=100, r_n=2.5, nu=0.4, rho=-0.7),
        fv, S0, mats,
    )
    iv200 = lifted_heston_iv_surface(
        LiftedHestonParams(H=0.10, n=200, r_n=2.5, nu=0.4, rho=-0.7),
        fv, S0, mats,
    )
    for T in mats:
        mask = np.isfinite(iv100[T]) & np.isfinite(iv200[T])
        diff = np.max(np.abs(iv100[T][mask] - iv200[T][mask]))
        assert diff < 1e-4, f"T={T}: n=100 vs n=200 max ΔIV = {diff:.2e}"
