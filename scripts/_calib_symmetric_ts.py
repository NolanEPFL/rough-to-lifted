"""Calibration worker for Extension 1 — the term-structure-split symmetric model
(9 params: w_S, w_L, a, H1, nu1, rho1, kappa2, nu2, rho2). Mirrors _calib_symmetric.py
(same DE/refine/loss/thinning/full-surface diagnostics); only the model differs.

Run as a script; writes one JSON to --outdir. No existing/tracked file modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import differential_evolution, minimize

import _symmetric_common as C
from src.two_factor_symmetric.term_structure_split import (
    SymmetricTwoFactorTSParams, ts_iv_surface, ts_weight,
)

PARAM_ORDER_TS = ["w_S", "w_L", "a", "H1", "nu1", "rho1", "kappa2", "nu2", "rho2"]
BOUNDS_TS = [
    (0.0, 1.0),      # w_S  — short-maturity block-2 share
    (0.0, 1.0),      # w_L  — long-maturity block-2 share
    (0.10, 20.0),    # a    — transition rate of the split
    (0.02, 0.49),    # H1
    (0.05, 3.00),    # nu1
    (-0.99, -0.01),  # rho1
    (0.05, 10.0),    # kappa2
    (0.01, 2.00),    # nu2
    (-0.99, -0.01),  # rho2
]


def make_params_ts(w_S, w_L, a, H1, nu1, rho1, kappa2, nu2, rho2):
    return SymmetricTwoFactorTSParams(
        w_S=w_S, w_L=w_L, a=a, H1=H1, nu1=nu1, rho1=rho1,
        kappa2=kappa2, nu2=nu2, rho2=rho2, n1=C.N1, r_n1=C.R_N1, n2=1)


def build_iv_ts(params, fv, S0, K, n_steps=200):
    return ts_iv_surface(params, fv, S0, K, n_steps=n_steps)


class ObjectiveTS:
    def __init__(self, fv, S0, K_calib, loss):
        self.fv = fv; self.S0 = S0; self.K_calib = K_calib; self.loss = loss

    def __call__(self, theta):
        try:
            p = make_params_ts(*theta)
            return float(self.loss(build_iv_ts(p, self.fv, self.S0, self.K_calib)))
        except Exception:
            return 1e6


def ev_diagnostics_ts(params, fv, Tmax, n=6000):
    tg = np.linspace(1e-4, Tmax, n)
    xg = np.array([float(fv(t)) for t in tg])
    w_t = ts_weight(tg, params.w_S, params.w_L, params.a)
    g1 = (1.0 - w_t) * xg; g2 = w_t * xg
    ev = g1 + g2
    return float(np.mean((g1 < 0) | (g2 < 0))), float(np.abs(ev / xg - 1.0).max())


def calibrate(train, test, lam, seed, workers, de_maxiter, de_popsize=10):
    L = C.build_losses(train, test, lam)
    obj = ObjectiveTS(L["fv_tr"], L["S0_tr"], L["K_calib"], L["loss_tr"])
    t0 = time.perf_counter()
    de = differential_evolution(obj, BOUNDS_TS, popsize=de_popsize, maxiter=de_maxiter,
                                seed=seed, polish=False, workers=workers, tol=1e-4,
                                updating="deferred" if workers != 1 else "immediate")
    best, fbest = de.x, float(de.fun)
    opt = minimize(obj, best, method="L-BFGS-B", bounds=BOUNDS_TS,
                   options={"maxiter": 200, "eps": 1e-4, "ftol": 1e-12})
    if float(opt.fun) < fbest:
        best, fbest = opt.x, float(opt.fun)
    elapsed = time.perf_counter() - t0

    par = make_params_ts(*best)
    pdict = {k: float(v) for k, v in zip(PARAM_ORDER_TS, best)}
    iv_is = build_iv_ts(par, L["fv_tr"], L["S0_tr"], L["K_tr"])
    iv_oos = build_iv_ts(par, L["fv_te"], L["S0_te"], L["K_te"])
    is_comp = {k: float(v) for k, v in L["loss_tr_diag"].components(iv_is).items()}
    oos_comp = {k: float(v) for k, v in L["loss_te_diag"].components(iv_oos).items()}
    surf, fv, _ = C.load_surface(train)
    Tmax = float(max(surf.ivs_per_T().keys()))
    frac_clip, max_rel = ev_diagnostics_ts(par, fv, Tmax)
    # record the calibrated w(t) profile on a few maturities
    w_profile = {f"{t:.2f}": float(ts_weight(t, par.w_S, par.w_L, par.a))
                 for t in (0.05, 0.25, 0.5, 1.0, min(2.0, Tmax))}
    return dict(train=train, test=test, lam=("inf" if np.isinf(lam) else lam),
                seed=seed, model="SYM-TS", params=pdict, bounds=[list(b) for b in BOUNDS_TS],
                param_order=PARAM_ORDER_TS, obj=fbest, in_sample=is_comp,
                temporal_oos=oos_comp, frac_clipped=float(frac_clip),
                max_relEV_dev=float(max_rel), w_profile=w_profile, elapsed_s=float(elapsed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--lam", default="0.0"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--de-maxiter", type=int, default=30)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default=str(C.ROOT / "results/09_symmetric/extension_ts"))
    a = ap.parse_args()
    lam = np.inf if a.lam.lower() == "inf" else float(a.lam)
    res = calibrate(a.train, a.test, lam, a.seed, a.workers, a.de_maxiter)
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / f"{a.tag}.json", "w"), indent=2)
    p = res["params"]
    print(f"[{a.tag}] obj={res['obj']:.6f} clip={res['frac_clipped']:.2%} "
          f"maxRelEV={res['max_relEV_dev']:.2e} iv_IS={res['in_sample']['iv_rmse']:.4f} "
          f"skew_IS={res['in_sample']['skew_rmse']:.4f} "
          f"w_S={p['w_S']:.3f} w_L={p['w_L']:.3f} a={p['a']:.2f} kappa2={p['kappa2']:.3f} "
          f"t={res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
