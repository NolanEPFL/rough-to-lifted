"""SPX volatility-surface loader and synthetic-surface generator.

Real data path
--------------
``load_spx_csv`` reads a CSV with columns:
    expiry_date, strike, bid, ask, option_type (C/P), quote_date, spot, risk_free_rate

Cleaning follows CONTEXT.md §9.2 (mid-price, OTM filter, IV inversion, etc.).
``fit_xi0_from_surface`` extracts the forward-variance curve ξ₀ from the ATM
total-variance term structure (practitioner standard, §9.2 step 7).

Synthetic path (no real data)
------------------------------
``make_synthetic_surface`` generates a surface from n=500 lifted Heston with
typical SPX-like parameters and small IV noise.  Used for Experiments 3 and 4
when no real CSV is available (CONTEXT.md §9.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from ..common.black_scholes import bs_vega, bs_implied_vol

IV_MIN, IV_MAX   = 0.01, 1.5
T_MIN            = 7.0 / 365.0
T_MAX            = 2.0
OOS_T_THRESHOLD  = 1.0

# Maturity-dependent log-moneyness ranges (Table 2 of Abi Jaber & Li 2025).
# Each entry: (T_upper_bound_in_years, k_min, k_max).
# The last row covers T >= 1 year.
_MONEYNESS_BRACKETS = [
    (14  / 365, -0.15,  0.03),
    (1   / 12,  -0.25,  0.03),
    (2   / 12,  -0.30,  0.04),
    (3   / 12,  -0.40,  0.15),
    (6   / 12,  -0.60,  0.15),
    (1.0,       -0.80,  0.20),
    (float("inf"), -1.50, 0.30),
]


def _moneyness_bounds(T: float) -> tuple[float, float]:
    for T_upper, k_min, k_max in _MONEYNESS_BRACKETS:
        if T < T_upper:
            return k_min, k_max
    return -1.50, 0.30


@dataclass
class SPXSurface:
    """Container for a single-date SPX surface, post-cleaning."""
    date:             str
    strikes:          np.ndarray    # (N,)
    maturities:       np.ndarray    # (N,)
    iv_mkt:           np.ndarray    # (N,)
    forwards:         np.ndarray    # (N,) forward per maturity
    weights:          np.ndarray    # (N,)
    oos_mask:         np.ndarray    # (N,) bool, True = OOS (T > 1y)
    xi0_grid:         Optional[np.ndarray] = None
    xi0_maturities:   Optional[np.ndarray] = None

    def strikes_per_T(self) -> dict[float, np.ndarray]:
        """dict {T → array of absolute strikes} for the calibration API."""
        out: dict[float, np.ndarray] = {}
        for T in np.unique(self.maturities):
            mask = self.maturities == T
            out[float(T)] = self.strikes[mask]
        return out

    def ivs_per_T(self) -> dict[float, np.ndarray]:
        """dict {T → array of market IVs}."""
        out: dict[float, np.ndarray] = {}
        for T in np.unique(self.maturities):
            mask = self.maturities == T
            out[float(T)] = self.iv_mkt[mask]
        return out

    def forward_per_T(self) -> dict[float, float]:
        """dict {T → forward price} (one forward per maturity)."""
        out: dict[float, float] = {}
        for T in np.unique(self.maturities):
            mask = self.maturities == T
            out[float(T)] = float(self.forwards[mask][0])
        return out


# ── Real SPX loader ──────────────────────────────────────────────────────────

def load_spx_csv(
    path: Path,
    weight_scheme: Literal["equal", "vega", "spread"] = "equal",
    oos_threshold: float = OOS_T_THRESHOLD,
) -> SPXSurface:
    """Load and clean an SPX surface CSV.

    Expected CSV columns (CONTEXT.md §9.1):
        expiry_date, strike, bid, ask, option_type, quote_date, spot, risk_free_rate

    Cleaning steps:
        1. Mid-price = (bid + ask) / 2.
        2. Filter: bid > 0, ask > 0, spread < 50% of mid, T in [T_MIN, T_MAX],
           log-moneyness in [LOG_MONEYNESS_MIN, LOG_MONEYNESS_MAX].
        3. Forward per maturity via put-call parity.
        4. OTM only (calls K > F, puts K < F).
        5. IV inversion via Brent.
        6. IV bounds filter.
        7. Weights.
    """
    import pandas as pd
    from datetime import datetime

    df = pd.read_csv(path)
    required = {"expiry_date", "strike", "bid", "ask", "option_type",
                "quote_date", "spot", "risk_free_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Parse dates → maturity in years
    quote_dt  = pd.to_datetime(df["quote_date"])
    expiry_dt = pd.to_datetime(df["expiry_date"])
    df["T"]   = (expiry_dt - quote_dt).dt.days / 365.0
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Basic option filters
    df = df[
        (df["bid"] > 0) & (df["ask"] > 0)
        & ((df["ask"] - df["bid"]) < 0.5 * df["mid"])
        & (df["T"] >= T_MIN) & (df["T"] <= T_MAX)
    ].copy()

    S0 = float(df["spot"].iloc[0])
    r  = float(df["risk_free_rate"].iloc[0]) if "risk_free_rate" in df.columns else 0.0

    # Forward per maturity via put-call parity
    out_rows = []
    for T, grp in df.groupby("T"):
        F = _extract_forward(grp, T, r, S0)
        # Maturity-dependent log-moneyness filter (Abi Jaber & Li 2025, Table 2)
        k_min, k_max = _moneyness_bounds(float(T))
        k = np.log(grp["strike"].values / F)
        keep = (k >= k_min) & (k <= k_max)
        # OTM only
        calls = (grp["option_type"].str.upper() == "C") & (grp["strike"].values > F)
        puts  = (grp["option_type"].str.upper() == "P") & (grp["strike"].values < F)
        otm   = calls | puts
        mask  = keep & otm
        sub   = grp[mask].copy()
        for _, row in sub.iterrows():
            otype = str(row["option_type"]).upper()
            iv    = bs_implied_vol(
                float(row["mid"]), F, float(row["strike"]), float(T),
                option_type=otype,
            )
            if not np.isfinite(iv) or iv < IV_MIN or iv > IV_MAX:
                continue
            out_rows.append({
                "T": T, "K": float(row["strike"]), "iv": iv, "F": F,
                "mid": float(row["mid"]),
            })

    if not out_rows:
        raise ValueError("No valid options after filtering.")

    res  = pd.DataFrame(out_rows)
    T_arr = res["T"].values
    K_arr = res["K"].values
    iv_arr = res["iv"].values
    F_arr  = res["F"].values

    # Weights
    if weight_scheme == "equal":
        w = np.ones(len(res))
    elif weight_scheme == "vega":
        w = np.array([bs_vega(f, k, t, iv) for f, k, t, iv in
                      zip(F_arr, K_arr, T_arr, iv_arr)])
        w = np.maximum(w, 1e-10)
    elif weight_scheme == "spread":
        spread = res["mid"].values * 0.1  # placeholder
        w = 1.0 / np.maximum(spread, 1e-6)
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    oos_mask = T_arr > oos_threshold
    date_str  = str(df["quote_date"].iloc[0])

    return SPXSurface(
        date=date_str,
        strikes=K_arr, maturities=T_arr,
        iv_mkt=iv_arr, forwards=F_arr,
        weights=w, oos_mask=oos_mask,
    )


def _extract_forward(
    grp: "pd.DataFrame", T: float, r: float, S0_fallback: float,
    n_pairs: int = 5,
) -> float:
    """Estimate F_T via put-call parity using the median over n_pairs near-ATM pairs.

    For each matched (C, P) pair at the same strike K_j:
        F_T^(j) = K_j + e^(rT) * (C_mid(K_j) - P_mid(K_j))

    The n_pairs strikes with the smallest |C_mid - P_mid| (i.e. closest to ATM)
    are selected and the median F_T^(j) is returned.  The median is more robust
    than a single-pair estimate when individual quotes are stale or noisy.
    """
    calls = grp[grp["option_type"].str.upper() == "C"]
    puts  = grp[grp["option_type"].str.upper() == "P"]
    if calls.empty or puts.empty:
        return S0_fallback * np.exp(r * T)
    merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if merged.empty:
        return S0_fallback * np.exp(r * T)

    # F_T implied by each matched pair
    er = np.exp(r * T)
    F_implied = (merged["strike"].values
                 + er * (merged["mid_c"].values - merged["mid_p"].values))

    # Select the n_pairs pairs closest to ATM (smallest |C - P|)
    diff = np.abs(merged["mid_c"].values - merged["mid_p"].values)
    top  = min(n_pairs, len(diff))
    idx  = np.argpartition(diff, top - 1)[:top]

    return float(np.median(F_implied[idx]))


# ── ξ₀ extraction ───────────────────────────────────────────────────────────

def fit_xi0_from_surface(
    surface: SPXSurface,
    method: Literal["atm_total_var", "vix_term"] = "atm_total_var",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ξ₀(t) from the ATM total-variance term structure.

    Returns (xi0_maturities, xi0_values).  Both are strictly positive.

    Method "atm_total_var" (CONTEXT.md §9.2 step 7):
        w(T) = σ²_ATM(T) · T  →  ξ₀(T) = dw/dT

    The PCHIP spline fitted to w(T) is differentiable.  Any negative
    derivatives are clipped to ε = 1e-4.
    """
    from scipy.interpolate import PchipInterpolator

    mats = np.unique(surface.maturities)
    atm_var = []

    for T in mats:
        mask = surface.maturities == T
        K_T  = surface.strikes[mask]
        iv_T = surface.iv_mkt[mask]
        F_T  = surface.forwards[mask][0]

        # ATM IV: interpolate to k=0 (log-moneyness=0)
        k    = np.log(K_T / F_T)
        idx  = np.argsort(np.abs(k))[:2]
        if len(idx) < 2:
            sigma_atm = iv_T[np.argmin(np.abs(k))]
        else:
            i1, i2 = int(idx[0]), int(idx[1])
            # Linear interpolation to k=0
            if abs(k[i2] - k[i1]) < 1e-10:
                sigma_atm = iv_T[i1]
            else:
                sigma_atm = iv_T[i1] + (0 - k[i1]) / (k[i2] - k[i1]) * (iv_T[i2] - iv_T[i1])
        atm_var.append(float(sigma_atm ** 2 * T))

    w_arr = np.array(atm_var)          # total-variance term structure w(T)
    pchip = PchipInterpolator(mats, w_arr)

    # ξ₀(T) = dw/dT, evaluated at a fine grid
    t_fine = np.linspace(mats[0], mats[-1], max(10 * len(mats), 50))
    xi0    = np.maximum(pchip.derivative()(t_fine), 1e-4)

    return t_fine, xi0


# ── Synthetic surface generator ──────────────────────────────────────────────

def make_synthetic_surface(
    H: float   = 0.10,
    nu: float  = 0.4,
    rho: float = -0.7,
    V0: float  = 0.04,
    S0: float  = 100.0,
    maturities: list[float] | None = None,
    n_strikes:  int   = 13,
    seed:       int   = 42,
    noise_std:  float = 0.001,
    n_lh:       int   = 500,
) -> SPXSurface:
    """Generate a synthetic SPX-like surface from n=500 lifted Heston.

    Parameters
    ----------
    noise_std : IV noise in vol-point units (default 10 bps) to simulate
                realistic market microstructure.
    n_lh      : number of LH factors for the ground-truth surface.

    Returns an SPXSurface with xi0_grid and xi0_maturities set.
    """
    from ..lifted_heston.params import LiftedHestonParams
    from ..lifted_heston.pricing import lifted_heston_iv_surface
    from ..common.forward_variance import FlatForwardVariance

    if maturities is None:
        maturities = [0.083, 0.25, 0.5, 1.0, 1.5, 2.0]

    fv         = FlatForwardVariance(V0)
    log_k_grid = np.linspace(-0.3, 0.2, n_strikes)
    strikes_arr = S0 * np.exp(log_k_grid)
    strikes_per_T = {T: strikes_arr for T in maturities}

    params = LiftedHestonParams(H=H, n=n_lh, r_n=2.5, nu=nu, rho=rho)
    iv_surface = lifted_heston_iv_surface(params, fv, S0, strikes_per_T)

    rng = np.random.default_rng(seed)

    T_list, K_list, iv_list, F_list = [], [], [], []
    for T in maturities:
        ivs_T = iv_surface[T]
        if noise_std > 0:
            ivs_T = ivs_T + rng.normal(0.0, noise_std, size=ivs_T.shape)
        for K, iv in zip(strikes_arr, ivs_T):
            if not np.isfinite(iv) or iv < IV_MIN or iv > IV_MAX:
                continue
            T_list.append(T)
            K_list.append(K)
            iv_list.append(iv)
            F_list.append(S0)   # zero rates: F = S0

    T_arr   = np.array(T_list)
    K_arr   = np.array(K_list)
    iv_arr  = np.array(iv_list)
    F_arr   = np.array(F_list)
    oos     = T_arr > OOS_T_THRESHOLD

    return SPXSurface(
        date="synthetic",
        strikes=K_arr, maturities=T_arr,
        iv_mkt=iv_arr, forwards=F_arr,
        weights=np.ones(len(T_arr)),
        oos_mask=oos,
        xi0_grid=np.full(100, V0),
        xi0_maturities=np.linspace(0.0, 2.0, 100),
    )
