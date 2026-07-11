"""Symmetric-split 7-parameter fair-run calibration worker (one config → one JSON).

Mirrors diagnostics_09/_calib8.py: same DE (popsize 10, maxiter 30, seed, tol 1e-4,
polish=False) + L-BFGS-B refinement (maxiter 200, eps 1e-4, ftol 1e-12), same
loss/thinning/weighting, FULL-surface iv_IS/iv_OOS diagnostics. Calibration prices
at n_steps=200; metrics at n_steps=200 (apples-to-apples with the committed
baselines and Table 8.6's clean two-factor column).

Run as a script so DE workers>1 multiprocessing has a __main__ guard.
Reproduces no committed result; writes only under results/09_symmetric/.
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


def calibrate(train, test, lam, seed, workers, de_maxiter, de_popsize=10):
    L = C.build_losses(train, test, lam)
    obj = C.Objective(L["fv_tr"], L["S0_tr"], L["K_calib"], L["loss_tr"])
    t0 = time.perf_counter()
    de = differential_evolution(obj, C.BOUNDS, popsize=de_popsize, maxiter=de_maxiter,
                                seed=seed, polish=False, workers=workers, tol=1e-4,
                                updating="deferred" if workers != 1 else "immediate")
    best, fbest = de.x, float(de.fun)
    opt = minimize(obj, best, method="L-BFGS-B", bounds=C.BOUNDS,
                   options={"maxiter": 200, "eps": 1e-4, "ftol": 1e-12})
    if float(opt.fun) < fbest:
        best, fbest = opt.x, float(opt.fun)
    elapsed = time.perf_counter() - t0

    par = C.make_params(*best)
    pdict = {k: float(v) for k, v in zip(C.PARAM_ORDER, best)}
    iv_is = C.build_iv(par, L["fv_tr"], L["S0_tr"], L["K_tr"])
    iv_oos = C.build_iv(par, L["fv_te"], L["S0_te"], L["K_te"])
    is_comp = {k: float(v) for k, v in L["loss_tr_diag"].components(iv_is).items()}
    oos_comp = {k: float(v) for k, v in L["loss_te_diag"].components(iv_oos).items()}

    # Tmax over the traded window, for the E[V]/ξ0 diagnostic.
    surf, fv, _ = C.load_surface(train)
    Tmax = float(max(surf.ivs_per_T().keys()))
    frac_clip, max_rel = C.ev_diagnostics(par, fv, Tmax)

    return dict(train=train, test=test, lam=("inf" if np.isinf(lam) else lam),
                seed=seed, workers=workers, de_maxiter=de_maxiter,
                params=pdict, bounds=[list(b) for b in C.BOUNDS],
                param_order=C.PARAM_ORDER,
                obj=fbest, in_sample=is_comp, temporal_oos=oos_comp,
                frac_clipped=float(frac_clip), max_relEV_dev=float(max_rel),
                elapsed_s=float(elapsed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--lam", default="0.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--de-maxiter", type=int, default=30)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default=str(C.ROOT / "results/09_symmetric"))
    a = ap.parse_args()
    lam = np.inf if a.lam.lower() == "inf" else float(a.lam)
    res = calibrate(a.train, a.test, lam, a.seed, a.workers, a.de_maxiter)
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / f"{a.tag}.json", "w"), indent=2)
    p = res["params"]
    print(f"[{a.tag}] obj={res['obj']:.6f} clip={res['frac_clipped']:.2%} "
          f"maxRelEV={res['max_relEV_dev']:.2e} iv_IS={res['in_sample']['iv_rmse']:.4f} "
          f"w={p['w']:.3f} kappa2={p['kappa2']:.3f} H1={p['H1']:.3f} "
          f"rho1={p['rho1']:.3f} rho2={p['rho2']:.3f} t={res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
