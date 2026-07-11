"""Shared harness for the symmetric-split two-factor fair run (Step 4).

Mirrors diagnostics_09/_tf8common.py EXACTLY for surface loading, thinning,
loss construction, the λ=∞ 5-strike grid, the full-surface vega diagnostic loss,
and the SVI skew curve (rmse_max=0.03) — the only change is the model: the
symmetric-split 7-parameter vector (w,H1,nu1,rho1,kappa2,nu2,rho2) replaces the
8-parameter decoupled CIR vector. This keeps the fair comparison apples-to-apples
with Table 8.6 (clean two-factor iv_IS on the FULL train surface vs committed
single-factor iv_IS).

NEW WORK; no existing/tracked file is modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.two_factor_symmetric.params import SymmetricTwoFactorParams
from src.two_factor_symmetric.pricing import symmetric_iv_surface

DATE_PAIRS = {
    "2024-08-05": "2024-08-06",
    "2024-12-04": "2024-12-05",
    "2025-04-03": "2025-04-04",
}
SEEDS = [42, 1, 2, 3, 4]
CONFIGS = [(d, lt) for d in DATE_PAIRS for lt in ("0", "inf")]

# 7-parameter calibration vector (erratum §8.9.7 spec). FIXED bounds — the
# additive split makes positivity automatic, so NO per-date positivity cap.
PARAM_ORDER = ["w", "H1", "nu1", "rho1", "kappa2", "nu2", "rho2"]
BOUNDS = [
    (0.0, 1.0),       # w      — block-2 variance share (convex weight)
    (0.02, 0.49),     # H1
    (0.05, 3.00),     # nu1
    (-0.99, -0.01),   # rho1
    (0.05, 10.0),     # kappa2 — block-2 OU speed (allow genuinely slow values)
    (0.01, 2.00),     # nu2
    (-0.99, -0.01),   # rho2
]
N1, R_N1 = 20, 2.5

RES07 = ROOT / "results/07_temporal_oos"
RES07B = ROOT / "results/07b_temporal_oos_two_factor"


# ── surface / xi0 (verbatim from _tf8common.load_surface / thin) ───────────────

def load_surface(date: str):
    surface = load_spx_csv(ROOT / "data" / f"spx_{date}.csv")
    xi0_m, xi0_v = fit_xi0_from_surface(surface)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surface.forwards.mean())
    return surface, fv, S0


def thin(market_ivs, strikes_per, forwards_per, max_mats, max_strikes):
    all_T = np.array(sorted(market_ivs.keys()))
    if max_mats and len(all_T) > max_mats:
        targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), max_mats))
        idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
        all_T = all_T[sorted(set(idx))]
    new_iv, new_K, new_F = {}, {}, {}
    for T in all_T:
        ivs = market_ivs[T]; Ks = strikes_per[T]
        if max_strikes and len(Ks) > max_strikes:
            idx = np.round(np.linspace(0, len(Ks) - 1, max_strikes)).astype(int)
            Ks = Ks[idx]; ivs = ivs[idx]
        new_iv[float(T)] = ivs
        new_K[float(T)] = Ks
        new_F[float(T)] = forwards_per.get(float(T), 0.0)
    return new_iv, new_K, new_F


# ── parameters / pricing ───────────────────────────────────────────────────────

def make_params(w, H1, nu1, rho1, kappa2, nu2, rho2):
    return SymmetricTwoFactorParams(
        w=w, H1=H1, nu1=nu1, rho1=rho1, kappa2=kappa2, nu2=nu2, rho2=rho2,
        n1=N1, r_n1=R_N1, n2=1)


def build_iv(params, fv, S0, K, n_steps=200):
    return symmetric_iv_surface(params, fv, S0, K, n_steps=n_steps)


# ── E[V_t] / clip diagnostics (no clip exists; verify exactly) ─────────────────

def xi_grid(fv, Tmax, n=6000):
    tg = np.linspace(1e-4, Tmax, n)
    xg = np.array([float(fv(t)) for t in tg])
    return tg, xg


def ev_diagnostics(params, fv, Tmax, n=6000):
    """frac_clipped and max|E[V_t]/ξ0 - 1| for the additive split.

    E[V_t] = (1-w)ξ0(t) + w ξ0(t) = ξ0(t) exactly; both forcings ≥ 0 so NO floor
    is ever applied. Returns (frac_clipped, max_rel_dev) — frac_clipped is the
    fraction of grid points where either forcing would need flooring (= 0).
    """
    tg, xg = xi_grid(fv, Tmax, n)
    s1, s2 = 1.0 - params.w, params.w
    g1 = s1 * xg
    g2 = s2 * xg
    ev = g1 + g2
    frac_clipped = float(np.mean((g1 < 0.0) | (g2 < 0.0)))
    rel = np.abs(ev / xg - 1.0)
    return frac_clipped, float(rel.max())


# ── loss construction (verbatim from _tf8common.build_losses) ──────────────────

def build_losses(train, test, lam):
    surf_tr, fv_tr, S0_tr = load_surface(train)
    mkt_tr, K_tr, F_tr = surf_tr.ivs_per_T(), surf_tr.strikes_per_T(), surf_tr.forward_per_T()
    mkt_c, K_c, F_c = thin(mkt_tr, K_tr, F_tr, 10, 20)
    if np.isinf(lam):
        calib_K = {T: np.array([F * 0.96, F * 0.98, F, F * 1.02, F * 1.04])
                   for T, F in F_c.items() if T in mkt_c}
        calib_mkt = {}
        for T, K5 in calib_K.items():
            kf = np.log(K_c[T] / F_c[T]); k5 = np.log(K5 / F_c[T]); o = np.argsort(kf)
            calib_mkt[T] = np.interp(k5, kf[o], mkt_c[T][o])
        loss_tr = IVSurfaceLoss(market_ivs=calib_mkt, strikes=calib_K, S0=S0_tr,
                                forwards=F_c, weights="equal", skew_lambda=np.inf,
                                in_sample_T_max=None)
        K_calib = calib_K
    else:
        loss_tr = IVSurfaceLoss(market_ivs=mkt_c, strikes=K_c, S0=S0_tr,
                                forwards=F_c, weights="vega", skew_lambda=lam,
                                in_sample_T_max=None)
        K_calib = K_c
    loss_tr_diag = IVSurfaceLoss(market_ivs=mkt_tr, strikes=K_tr, S0=S0_tr,
                                 forwards=F_tr, weights="vega", skew_lambda=0.0,
                                 in_sample_T_max=None)
    surf_te, fv_te, S0_te = load_surface(test)
    mkt_te, K_te, F_te = surf_te.ivs_per_T(), surf_te.strikes_per_T(), surf_te.forward_per_T()
    loss_te_diag = IVSurfaceLoss(market_ivs=mkt_te, strikes=K_te, S0=S0_te,
                                 forwards=F_te, weights="vega", skew_lambda=0.0,
                                 in_sample_T_max=None)
    return dict(fv_tr=fv_tr, S0_tr=S0_tr, loss_tr=loss_tr, K_calib=K_calib,
                loss_tr_diag=loss_tr_diag, K_tr=K_tr,
                fv_te=fv_te, S0_te=S0_te, loss_te_diag=loss_te_diag, K_te=K_te)


# ── picklable DE objective (module-level → safe for workers>1) ─────────────────

class Objective:
    def __init__(self, fv, S0, K_calib, loss):
        self.fv = fv; self.S0 = S0; self.K_calib = K_calib; self.loss = loss

    def __call__(self, theta):
        try:
            p = make_params(*theta)
            ivs = build_iv(p, self.fv, self.S0, self.K_calib)
            return float(self.loss(ivs))
        except Exception:
            return 1e6


# ── SVI ATM-skew term structure (verbatim from _tf8common.svi_skew_curve) ──────

def svi_skew_curve(iv_dict, K_dict, F_dict, rmse_max=0.03):
    """Return {T: dσ/dk|_{k=0}} via SVI, dropping slices with SVI RMSE>rmse_max."""
    from src.data.svi import fit_svi_from_iv_dict
    fits = fit_svi_from_iv_dict(iv_dict, K_dict, F_dict)
    out = {}
    for T, f in fits.items():
        if f.rmse_iv <= rmse_max:
            out[float(T)] = float(f.atm_skew_derivative())
    return out


# ── block-2 activity diagnostics (analogue of _phase4.block2_activity) ─────────

def block2_activity(date, params_dict):
    """Leading-order ATM-skew ratio |K2/K1| (eq. 6.25, g0^(2)=w ξ0) and the
    SVI block-2-on minus block-2-off skew difference over the traded window."""
    surf, fv, S0 = load_surface(date)
    K_full = surf.strikes_per_T(); F_full = surf.forward_per_T()
    p = params_dict
    xi0_0 = float(fv(0.0))
    par = make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], p["nu2"], p["rho2"])
    # block 2 OFF: kill its stochastic-vol skew contribution (nu2 → tiny)
    par_off = make_params(p["w"], p["H1"], p["nu1"], p["rho1"], p["kappa2"], 1e-8, p["rho2"])
    sum_c1_over_x1 = float(np.sum(par.c1 / par.x1))
    V1_0 = (1.0 - p["w"]) * xi0_0
    V2_0 = p["w"] * xi0_0
    K1 = p["rho1"] * p["nu1"] * V1_0 * sum_c1_over_x1
    K2 = p["rho2"] * p["nu2"] * V2_0 / p["kappa2"]   # single factor: Σ c/x = 1/κ2
    sk_full = svi_skew_curve(build_iv(par, fv, S0, K_full, n_steps=400), K_full, F_full)
    sk_off = svi_skew_curve(build_iv(par_off, fv, S0, K_full, n_steps=400), K_full, F_full)
    common = sorted(set(sk_full) & set(sk_off))
    diff = np.array([sk_full[T] - sk_off[T] for T in common])
    absK = abs(K2 / K1) if K1 != 0 else float("nan")
    return dict(absK2K1=float(absK),
                b2_skew_max=(float(np.max(np.abs(diff))) if len(diff) else float("nan")),
                b2_skew_med=(float(np.median(np.abs(diff))) if len(diff) else float("nan")),
                drho=abs(p["rho1"] - p["rho2"]), kappa2=p["kappa2"], w=p["w"])


# ── committed baselines (verbatim from _phase4.single_factor_iv / committed_tf) ─

def single_factor_iv(date, test, lt):
    """Best single-factor IS/tOOS iv_rmse from committed 07 results (same ξ0)."""
    f = RES07 / f"{date}_{test}_42_lam{('inf' if lt == 'inf' else '0.0')}/temporal_oos_results.json"
    if not f.exists():
        return None
    R = json.load(open(f))["results"]
    best = min(R.items(), key=lambda kv: kv[1]["in_sample"]["iv_rmse"])
    return dict(model=best[0],
                iv_IS=best[1]["in_sample"]["iv_rmse"],
                iv_tOOS=best[1]["temporal_oos"]["iv_rmse"],
                all_iv_IS={k: v["in_sample"]["iv_rmse"] for k, v in R.items()})


def committed_tf_iv(date, lt):
    f = RES07B / f"{date}_{DATE_PAIRS[date]}_42_lam{('inf' if lt == 'inf' else '0.0')}/temporal_oos_results.json"
    if not f.exists():
        return None
    R = json.load(open(f))["results"]["TF-LH"]
    return dict(iv_IS=R["in_sample"]["iv_rmse"], iv_tOOS=R["temporal_oos"]["iv_rmse"])
