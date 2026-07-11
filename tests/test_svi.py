"""Tests for src/data/svi.py — raw SVI slice fitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.svi import SVISliceFit, fit_svi_slice, fit_svi_surface, fit_svi_from_iv_dict

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_SPX_CSV  = _DATA_DIR / "spx_2024-12-04.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _svi_w(k, a, b, rho, m, s):
    d = k - m
    return a + b * (rho * d + np.sqrt(d**2 + s**2))


# ── Test 1: recovery on clean synthetic SVI data ──────────────────────────────

def test_svi_recovery():
    """fit_svi_slice recovers rho and ATM IV from noiseless synthetic data."""
    T = 0.5
    a_true, b_true, rho_true, m_true, s_true = 0.04 * T, 0.4, -0.7, 0.0, 0.1

    k = np.linspace(-0.5, 0.5, 21)
    w = _svi_w(k, a_true, b_true, rho_true, m_true, s_true)
    iv = np.sqrt(w / T)

    fit = fit_svi_slice(k, iv, T)

    # rho recovery
    assert abs(fit.rho - rho_true) < 0.05, (
        f"rho_hat={fit.rho:.4f} differs from true {rho_true} by > 0.05"
    )

    # ATM IV recovery — true ATM IV from w(0)
    w0_true = _svi_w(np.array([0.0]), a_true, b_true, rho_true, m_true, s_true)[0]
    atm_iv_true = np.sqrt(w0_true / T)
    assert abs(fit.atm_iv() - atm_iv_true) < 1e-4, (
        f"atm_iv={fit.atm_iv():.6f} differs from true {atm_iv_true:.6f} by > 1e-4"
    )


# ── Test 2: quadratic smile ────────────────────────────────────────────────────

def test_svi_quadratic():
    """SVI fits a quadratic smile to better than 1e-3 RMSE (SVI contains quadratics).

    k range limited to [-0.2, 0.2] so the approximation error (O(k^4/s^3)) is
    small enough for s in [1e-4, 1.0] to achieve RMSE < 1e-3.
    """
    T = 0.5
    k = np.linspace(-0.2, 0.2, 21)
    # Quadratic total variance: w(k) = 0.04 + 0.1*k + 0.3*k^2
    w = 0.04 + 0.1 * k + 0.3 * k**2
    w = np.maximum(w, 1e-6)
    iv = np.sqrt(w / T)

    fit = fit_svi_slice(k, iv, T)
    assert fit.rmse_iv < 1e-3, f"rmse_iv={fit.rmse_iv:.5f} > 1e-3 on quadratic smile"


# ── Test 3: real-data RMSE sanity ─────────────────────────────────────────────

@pytest.mark.skipif(not _SPX_CSV.exists(), reason="spx_2024-12-04.csv not found")
def test_svi_real_data_rmse():
    """Median RMSE < 50 bps and worst slice < 200 bps on 2024-12-04 SPX surface."""
    from src.data.spx_loader import load_spx_csv

    surf = load_spx_csv(_SPX_CSV)
    fits = fit_svi_surface(surf)
    assert len(fits) > 0, "No slices fitted"

    rmses = [f.rmse_iv for f in fits.values()]
    med = float(np.median(rmses))
    worst = float(np.max(rmses))

    assert med < 0.005, f"Median RMSE = {med:.4f} >= 0.005 (50 bps)"
    assert worst < 0.02, f"Worst RMSE = {worst:.4f} >= 0.02 (200 bps)"


# ── Test 4: sign of ATM skew derivative ───────────────────────────────────────

@pytest.mark.skipif(not _SPX_CSV.exists(), reason="spx_2024-12-04.csv not found")
def test_svi_skew_sign():
    """ATM skew derivative is negative for all T > 1/52 on SPX (downward skew)."""
    from src.data.spx_loader import load_spx_csv

    surf = load_spx_csv(_SPX_CSV)
    fits = fit_svi_surface(surf)

    T_min = 1.0 / 52
    for T, fit in fits.items():
        if T <= T_min:
            continue
        skew = fit.atm_skew_derivative()
        assert skew < 0, (
            f"T={T:.4f}: ATM skew derivative = {skew:.4f} is not negative"
        )


# ── Test 5: fit_svi_from_iv_dict API ─────────────────────────────────────────

def test_svi_from_iv_dict():
    """fit_svi_from_iv_dict returns correct keys and accurate ATM IV."""
    T1, T2 = 0.25, 1.0
    F1, F2 = 100.0, 100.0
    a, b, rho, m, s = 0.02, 0.3, -0.6, 0.0, 0.1

    # 7 log-moneyness strikes per slice
    k_grid = np.linspace(-0.3, 0.2, 7)
    K1 = F1 * np.exp(k_grid)
    K2 = F2 * np.exp(k_grid)

    def make_iv(T):
        w = _svi_w(k_grid, a * T, b, rho, m, s)
        return np.sqrt(np.maximum(w, 1e-8) / T)

    iv_dict = {T1: make_iv(T1), T2: make_iv(T2)}
    K_dict  = {T1: K1,          T2: K2}
    F_dict  = {T1: F1,          T2: F2}

    fits = fit_svi_from_iv_dict(iv_dict, K_dict, F_dict)

    assert set(fits.keys()) == {T1, T2}, "Keys do not match input maturities"

    for T, F in [(T1, F1), (T2, F2)]:
        w0_true = _svi_w(np.array([0.0]), a * T, b, rho, m, s)[0]
        atm_true = float(np.sqrt(max(w0_true, 0.0) / T))
        assert abs(fits[T].atm_iv() - atm_true) < 1e-3, (
            f"T={T}: atm_iv={fits[T].atm_iv():.5f} differs from true {atm_true:.5f} by > 1e-3"
        )
