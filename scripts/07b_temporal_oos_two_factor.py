"""scripts/07b_temporal_oos_two_factor.py — Temporal OOS for the two-factor LH.

Same protocol as 07_temporal_oos.py but for the two-factor lifted Heston
(Chapter 8, Case I: rho_12=0). Only COS pricing — no MC needed.

Calibrated parameters: H1, nu1, rho1, lam2, theta2, nu2, rho2
Fixed:                  n1=20, r_n1=2.5, V2_0=theta2 (stationary start)

Usage
-----
    # lambda=0 (IV-RMSE only)
    python scripts/07b_temporal_oos_two_factor.py --train 2024-08-05 --test 2024-08-06

    # lambda=inf (skew-only calibration)
    python scripts/07b_temporal_oos_two_factor.py --train 2024-08-05 --test 2024-08-06 --lam inf

    # Fast smoke test
    python scripts/07b_temporal_oos_two_factor.py --train 2024-08-05 --test 2024-08-06 \\
        --max-maturities 8 --max-strikes 15 --de-maxiter 10

Output
------
    results/07b_temporal_oos_two_factor/<train>_<test>_<seed>_lam<lam>/
        temporal_oos_results.json
        summary_table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.two_factor_lifted_heston.params import TwoFactorLiftedHestonParams
from src.two_factor_lifted_heston.pricing import (
    two_factor_lh_call_prices,
    two_factor_lh_iv_surface,
)


# ── Parameter bounds ──────────────────────────────────────────────────────────
# Order: [nu1, rho1, H1, lam2, theta2, nu2, rho2]
TF_BOUNDS = [
    (0.05, 3.00),   # nu1    vol-of-vol block 1
    (-0.99, -0.01), # rho1   leverage block 1
    (0.02, 0.49),   # H1     Hurst index
    (0.10, 10.0),   # lam2   mean-reversion speed block 2
    (0.001, 0.04),  # theta2 long-run variance block 2
    (0.01, 2.00),   # nu2    vol-of-vol block 2
    (-0.99, -0.01), # rho2   leverage block 2
]
N1, R_N1 = 20, 2.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_surface(date: str, data_dir: Path):
    csv = data_dir / f"spx_{date}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found.")
    surface = load_spx_csv(csv)
    xi0_m, xi0_v = fit_xi0_from_surface(surface)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surface.forwards.mean())
    return surface, fv, S0


def _thin(market_ivs, strikes_per, forwards_per, max_mats, max_strikes):
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
        new_K[float(T)]  = Ks
        new_F[float(T)]  = forwards_per.get(float(T), 0.0)
    return new_iv, new_K, new_F


def _make_params(nu1, rho1, H1, lam2, theta2, nu2, rho2):
    return TwoFactorLiftedHestonParams(
        H1=H1, n1=N1, r_n1=R_N1, nu1=nu1, rho1=rho1,
        lam2=lam2, theta2=theta2, nu2=nu2, rho2=rho2,
        V2_0=theta2,   # stationary initial condition
    )


def _build_iv_surface(params, fv, S0, strikes_per_T):
    return two_factor_lh_iv_surface(params, fv, S0, strikes_per_T)


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: None if x != x else x)
    tmp.replace(path)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_two_factor_lh(
    loss: IVSurfaceLoss,
    fv: PiecewiseConstantForwardVariance,
    seed: int = 42,
    de_popsize: int = 10,
    de_maxiter: int = 30,
    refine: bool = True,
):
    S0 = loss.S0
    strikes_per = loss.strikes
    evals = [0]

    def obj(theta):
        nu1, rho1, H1, lam2, theta2, nu2, rho2 = theta
        evals[0] += 1
        try:
            params = _make_params(nu1, rho1, H1, lam2, theta2, nu2, rho2)
            ivs = _build_iv_surface(params, fv, S0, strikes_per)
            return loss(ivs)
        except Exception:
            return 1e6

    t0 = time.perf_counter()
    de_res = differential_evolution(
        obj, TF_BOUNDS,
        popsize=de_popsize, maxiter=de_maxiter,
        seed=seed, polish=False, workers=1, tol=1e-4,
    )
    best = de_res.x

    if refine:
        opt = minimize(
            obj, x0=best, method="L-BFGS-B", bounds=TF_BOUNDS,
            options={"maxiter": 200, "eps": 1e-4, "ftol": 1e-12},
        )
        if opt.fun < de_res.fun:
            best = opt.x

    nu1, rho1, H1, lam2, theta2, nu2, rho2 = best
    params_best = _make_params(nu1, rho1, H1, lam2, theta2, nu2, rho2)
    return {
        "nu1":    float(nu1),
        "rho1":   float(rho1),
        "H1":     float(H1),
        "lam2":   float(lam2),
        "theta2": float(theta2),
        "nu2":    float(nu2),
        "rho2":   float(rho2),
        "V2_0":   float(theta2),
    }, params_best, evals[0], time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          required=True)
    parser.add_argument("--test",           required=True)
    parser.add_argument("--lam",            type=str, default="0.0")
    parser.add_argument("--de-maxiter",     type=int, default=30)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--max-maturities", type=int, default=10)
    parser.add_argument("--max-strikes",    type=int, default=20)
    parser.add_argument("--data-dir",       type=Path, default=_ROOT / "data")
    parser.add_argument("--out",            type=Path,
                        default=_ROOT / "results/07b_temporal_oos_two_factor")
    args = parser.parse_args()

    lam = np.inf if args.lam.lower() == "inf" else float(args.lam)
    lam_tag = "inf" if np.isinf(lam) else args.lam
    out_dir = args.out / f"{args.train}_{args.test}_{args.seed}_lam{lam_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load surfaces ──────────────────────────────────────────────────────
    print(f"Loading train surface: {args.train} ...")
    surf_tr, fv_tr, S0_tr = _load_surface(args.train, args.data_dir)
    mkt_tr = surf_tr.ivs_per_T()
    K_tr   = surf_tr.strikes_per_T()
    F_tr   = surf_tr.forward_per_T()

    print(f"Loading test  surface: {args.test}  ...")
    surf_te, fv_te, S0_te = _load_surface(args.test, args.data_dir)
    mkt_te = surf_te.ivs_per_T()
    K_te   = surf_te.strikes_per_T()
    F_te   = surf_te.forward_per_T()

    # ── 2. Thin train surface ─────────────────────────────────────────────────
    max_mats    = args.max_maturities or len(mkt_tr)
    max_strikes = args.max_strikes    or 9999
    mkt_tr_c, K_tr_c, F_tr_c = _thin(mkt_tr, K_tr, F_tr, max_mats, max_strikes)
    print(f"Train: {len(mkt_tr_c)} mats × {max_strikes} strikes = "
          f"{sum(len(v) for v in mkt_tr_c.values())} calib points")
    print(f"Test:  {len(mkt_te)} mats, {sum(len(v) for v in mkt_te.values())} points")

    # ── 3. Build calibration loss ─────────────────────────────────────────────
    if np.isinf(lam):
        # 5-strike skew grid per maturity (same shortcut as 07_temporal_oos)
        calib_K = {
            T: np.array([F * 0.96, F * 0.98, F * 1.00, F * 1.02, F * 1.04])
            for T, F in F_tr_c.items() if T in mkt_tr_c
        }
        calib_mkt = {}
        for T, K5 in calib_K.items():
            K_full = K_tr_c[T]; iv_full = mkt_tr_c[T]; F = F_tr_c[T]
            k_full = np.log(K_full / F); k5 = np.log(K5 / F)
            order  = np.argsort(k_full)
            calib_mkt[T] = np.interp(k5, k_full[order], iv_full[order])
        loss_tr = IVSurfaceLoss(
            market_ivs=calib_mkt, strikes=calib_K, S0=S0_tr,
            forwards=F_tr_c, weights="equal", skew_lambda=np.inf,
            in_sample_T_max=None,
        )
        K_calib = calib_K
        print(f"lam=inf: 5-strike skew grid, {len(calib_K)*5} calib points")
    else:
        loss_tr = IVSurfaceLoss(
            market_ivs=mkt_tr_c, strikes=K_tr_c, S0=S0_tr,
            forwards=F_tr_c, weights="vega", skew_lambda=lam,
            in_sample_T_max=None,
        )
        K_calib = K_tr_c

    # Diagnostic losses on FULL surfaces (no thinning)
    loss_tr_diag = IVSurfaceLoss(
        market_ivs=mkt_tr, strikes=K_tr, S0=S0_tr,
        forwards=F_tr, weights="vega", skew_lambda=0.0,
        in_sample_T_max=None,
    )
    loss_te_diag = IVSurfaceLoss(
        market_ivs=mkt_te, strikes=K_te, S0=S0_te,
        forwards=F_te, weights="vega", skew_lambda=0.0,
        in_sample_T_max=None,
    )

    # ── 4. Calibrate two-factor LH ────────────────────────────────────────────
    print(f"\n[TF-LH] Calibrating on {args.train} (lam={lam}) ...")
    p_dict, params_best, n_evals, elapsed = calibrate_two_factor_lh(
        loss_tr, fv_tr,
        seed=args.seed,
        de_popsize=10,
        de_maxiter=args.de_maxiter,
        refine=True,
    )

    # ── 5. Diagnostics on full surfaces ───────────────────────────────────────
    ivs_tr_full = _build_iv_surface(params_best, fv_tr, S0_tr, K_tr)
    ivs_te_full = _build_iv_surface(params_best, fv_te, S0_te, K_te)

    is_comp  = loss_tr_diag.components(ivs_tr_full)
    oos_comp = loss_te_diag.components(ivs_te_full)

    print(f"  params: H1={p_dict['H1']:.4f}  nu1={p_dict['nu1']:.4f}  "
          f"rho1={p_dict['rho1']:.4f}")
    print(f"          lam2={p_dict['lam2']:.3f}  theta2={p_dict['theta2']:.4f}  "
          f"nu2={p_dict['nu2']:.4f}  rho2={p_dict['rho2']:.4f}")
    print(f"  IS  : iv_rmse={is_comp['iv_rmse']:.4f}  "
          f"skew_rmse={is_comp['skew_rmse']:.4f}")
    print(f"  t-OOS: iv_rmse={oos_comp['iv_rmse']:.4f}  "
          f"skew_rmse={oos_comp['skew_rmse']:.4f}  t={elapsed:.0f}s")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "model":          "TF-LH",
            "train_date":     args.train,
            "test_date":      args.test,
            "lambda":         float("inf") if np.isinf(lam) else lam,
            "n1":             N1, "r_n1": R_N1,
            "V2_0_fixed_to":  "theta2",
            "seed":           args.seed,
            "max_maturities": args.max_maturities,
            "max_strikes":    args.max_strikes,
        },
        "results": {
            "TF-LH": {
                "params":       p_dict,
                "train_date":   args.train,
                "test_date":    args.test,
                "lambda":       float("inf") if np.isinf(lam) else lam,
                "in_sample":    is_comp,
                "temporal_oos": oos_comp,
                "n_evals":      n_evals,
                "elapsed_s":    elapsed,
            }
        },
    }
    _save(out_dir / "temporal_oos_results.json", output)

    # ── 7. Summary table ──────────────────────────────────────────────────────
    header = (f"\nTF-LH Temporal OOS: train={args.train}  test={args.test}  lam={lam}\n"
              f"{'Model':<8} {'H1':>6} {'rho1':>7} {'nu1':>6} "
              f"{'lam2':>6} {'th2':>6} {'nu2':>6} {'rho2':>7} "
              f"{'iv_IS':>8} {'sk_IS':>8} {'iv_tOOS':>9} {'sk_tOOS':>9}\n"
              + "-" * 90)
    p = p_dict
    line = (f"{'TF-LH':<8} {p['H1']:>6.3f} {p['rho1']:>7.3f} {p['nu1']:>6.3f} "
            f"{p['lam2']:>6.2f} {p['theta2']:>6.4f} {p['nu2']:>6.3f} {p['rho2']:>7.3f} "
            f"{is_comp['iv_rmse']:>8.4f} {is_comp['skew_rmse']:>8.4f} "
            f"{oos_comp['iv_rmse']:>9.4f} {oos_comp['skew_rmse']:>9.4f}")
    print(header)
    print(line)

    with open(out_dir / "summary_table.txt", "w") as f:
        f.write(header + "\n" + line + "\n")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
