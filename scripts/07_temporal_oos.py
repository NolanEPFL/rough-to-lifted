"""scripts/07_temporal_oos.py — Temporal out-of-sample evaluation.

Calibrates all four models on a TRAIN date using all maturities (no IS/OOS
maturity split), then evaluates the calibrated parameters on a TEST date by
re-extracting xi0 from the test surface and keeping (H, rho, nu/eta) fixed.

This tests whether calibrated parameters are stable across trading days.

Protocol
--------
1. Load train surface  →  extract xi0_train  →  calibrate (H, rho, nu/eta)
2. Load test  surface  →  extract xi0_test   →  price with train params + xi0_test
3. Report IS (train date) and temporal OOS (test date) IV-RMSE and skew-RMSE

Usage
-----
    # Full surface, default lambda=0
    python scripts/07_temporal_oos.py --train 2024-08-05 --test 2024-08-06

    # Fast smoke test with thinning
    python scripts/07_temporal_oos.py --train 2024-08-05 --test 2024-08-06 \\
        --max-maturities 8 --max-strikes 15 --de-maxiter 5 --M-paths 10000

    # Also test skew-regularised calibration
    python scripts/07_temporal_oos.py --train 2024-08-05 --test 2024-08-06 \\
        --lam 0.1

Output
------
    results/07_temporal_oos/<train>_<test>_<seed>/
        temporal_oos_results.json
        summary_table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.calibration.optimizer import calibrate_lifted_heston, calibrate_abergomi
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.kernel_fit import ABergomiParams
from src.abergomi.pricing import abergomi_iv_surface


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_surface(date: str, data_dir: Path):
    csv = data_dir / f"spx_{date}.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run extract_spx_eod_quotes.py --date {date} first."
        )
    surface = load_spx_csv(csv)
    xi0_mats, xi0_vals = fit_xi0_from_surface(surface)
    fv = PiecewiseConstantForwardVariance(xi0_mats, xi0_vals)
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


def _build_lh(p, n, r_n, kernel, fv, S0, strikes):
    params = LiftedHestonParams(H=p["H"], n=n, r_n=r_n,
                                 nu=p["nu"], rho=p["rho"], kernel=kernel)
    return lifted_heston_iv_surface(params, fv, S0, strikes)


def _build_ab(p, n, r_n, kernel, fv, S0, strikes, M_paths, seed):
    from src.calibration.optimizer import _build_ab_params
    params = _build_ab_params(p["H"], n, p["eta"], p["rho"], kernel, r_n)
    ivs, _ = abergomi_iv_surface(
        params, fv, S0, strikes, M_paths=M_paths, qmc=False, seed=seed)
    return ivs


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: None if x != x else x)
    tmp.replace(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          required=True, help="Train date YYYY-MM-DD")
    parser.add_argument("--test",           required=True, help="Test  date YYYY-MM-DD")
    parser.add_argument("--lam",            type=float, default=0.0,
                        help="skew_lambda for calibration (0=pure IV, inf=pure skew)")
    parser.add_argument("--n-factors",      type=int,   default=20)
    parser.add_argument("--r-n",            type=float, default=2.5)
    parser.add_argument("--M-paths",        type=int,   default=50_000)
    parser.add_argument("--de-maxiter",     type=int,   default=30)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--max-maturities", type=int,   default=8,
                        help="Log-spaced maturities for calibration (default 8, 0=all)")
    parser.add_argument("--max-strikes",    type=int,   default=15,
                        help="Strikes per maturity for calibration (default 15, 0=all)")
    parser.add_argument("--no-lh-l2",       action="store_true")
    parser.add_argument("--data-dir",       type=Path,
                        default=_ROOT / "data")
    parser.add_argument("--out",            type=Path,
                        default=_ROOT / "results/07_temporal_oos")
    args = parser.parse_args()

    lam = np.inf if str(args.lam).lower() == "inf" else float(args.lam)
    lam_tag = "inf" if np.isinf(lam) else str(lam)
    out_dir = args.out / f"{args.train}_{args.test}_{args.seed}_lam{lam_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    N, R_N = args.n_factors, args.r_n

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

    # ── 2. Optional thinning on train only ────────────────────────────────────
    max_mats    = args.max_maturities or len(mkt_tr)
    max_strikes = args.max_strikes    or 9999
    mkt_tr, K_tr, F_tr = _thin(mkt_tr, K_tr, F_tr, max_mats, max_strikes)
    print(f"Train surface: {len(mkt_tr)} maturities, "
          f"{sum(len(v) for v in mkt_tr.values())} points "
          f"(thinned from full surface, all T included)")

    print(f"Test  surface: {len(mkt_te)} maturities, "
          f"{sum(len(v) for v in mkt_te.values())} points")

    # ── 3. Build calibration grid and loss ───────────────────────────────────
    # For lambda=inf: only 5 strikes per maturity needed for 98/102 skew.
    # This is ~30x faster than pricing the full grid for no benefit.
    # OOS evaluation always uses the full test surface for a complete picture.
    if np.isinf(lam):
        calib_K = {
            T: np.array([F * 0.96, F * 0.98, F * 1.00, F * 1.02, F * 1.04])
            for T, F in F_tr.items() if T in mkt_tr
        }
        # Market IVs at the 5-strike grid (interpolated from full surface)
        calib_mkt = {}
        for T, K5 in calib_K.items():
            K_full = K_tr[T]; iv_full = mkt_tr[T]
            k_full = np.log(K_full / F_tr[T])
            k5     = np.log(K5     / F_tr[T])
            order  = np.argsort(k_full)
            calib_mkt[T] = np.interp(k5, k_full[order], iv_full[order])
        print(f"lam=inf: using 5-strike skew grid "
              f"({len(calib_K)} mats x 5 strikes = {len(calib_K)*5} pts)")
        loss_tr = IVSurfaceLoss(
            market_ivs=calib_mkt, strikes=calib_K, S0=S0_tr,
            forwards=F_tr, weights="equal",   # vega undefined on 5-pt grid
            skew_lambda=np.inf,
            in_sample_T_max=None,
        )
        K_calib = calib_K   # pricing grid used during calibration
    else:
        loss_tr = IVSurfaceLoss(
            market_ivs=mkt_tr, strikes=K_tr, S0=S0_tr,
            forwards=F_tr, weights="vega",
            skew_lambda=lam,
            in_sample_T_max=None,
        )
        K_calib = K_tr

    # OOS diagnostic loss — always full test surface, skew_lambda=0
    loss_te = IVSurfaceLoss(
        market_ivs=mkt_te, strikes=K_te, S0=S0_te,
        forwards=F_te, weights="vega",
        skew_lambda=0.0,
        in_sample_T_max=None,
    )
    # IS diagnostic loss — always full train surface regardless of calib grid
    loss_tr_diag = IVSurfaceLoss(
        market_ivs=mkt_tr, strikes=K_tr, S0=S0_tr,
        forwards=F_tr, weights="vega",
        skew_lambda=0.0,
        in_sample_T_max=None,
    )

    common_lh = dict(n=N, r_n=R_N, seed=args.seed,
                     de_popsize=10, de_maxiter=args.de_maxiter, refine=True)
    common_ab = dict(n=N, r_n=R_N, seed=args.seed,
                     M_paths=args.M_paths, de_popsize=10,
                     de_maxiter=args.de_maxiter, refine=True,
                     final_M_paths=min(args.M_paths * 4, 100_000))

    models = [("LH-geo", "lh", "geometric"),
              ("aB-L2",  "ab", "l2"),
              ("aB-geo", "ab", "geometric")]
    if not args.no_lh_l2:
        models.insert(1, ("LH-L2", "lh", "l2"))

    results = {}

    # ── 4. Calibrate on train, evaluate on test ───────────────────────────────
    for name, family, kernel in models:
        print(f"\n[{name}] Calibrating on {args.train} (lam={lam}) ...")
        t0 = time.perf_counter()

        if family == "lh":
            # Calibrate on calib grid (5-strike if lam=inf, full otherwise)
            res = calibrate_lifted_heston(loss_tr, fv_tr, kernel=kernel, **common_lh)
            p   = res.params
            # Diagnostics always on full surfaces
            ivs_tr_full = _build_lh(p, N, R_N, kernel, fv_tr, S0_tr, K_tr)
            ivs_te_full = _build_lh(p, N, R_N, kernel, fv_te, S0_te, K_te)
        else:
            res = calibrate_abergomi(loss_tr, fv_tr, kernel=kernel, **common_ab)
            p   = res.params
            ivs_tr_full = _build_ab(p, N, R_N, kernel, fv_tr, S0_tr, K_tr,
                                     args.M_paths * 4, args.seed + 10)
            ivs_te_full = _build_ab(p, N, R_N, kernel, fv_te, S0_te, K_te,
                                     args.M_paths * 4, args.seed + 20)

        elapsed = time.perf_counter() - t0

        # Diagnostics on full surfaces (IS = train day, OOS = test day)
        is_comp  = loss_tr_diag.components(ivs_tr_full)
        oos_comp = loss_te.components(ivs_te_full)

        results[name] = {
            "params":       p,
            "train_date":   args.train,
            "test_date":    args.test,
            "lambda":       float("inf") if np.isinf(lam) else lam,
            "in_sample":    is_comp,
            "temporal_oos": oos_comp,
            "n_evals":      res.n_evals,
            "elapsed_s":    elapsed,
        }

        print(f"  params: {p}")
        print(f"  IS  : iv_rmse={is_comp['iv_rmse']:.4f}  "
              f"skew_rmse={is_comp['skew_rmse']:.4f}")
        print(f"  t-OOS: iv_rmse={oos_comp['iv_rmse']:.4f}  "
              f"skew_rmse={oos_comp['skew_rmse']:.4f}  t={elapsed:.0f}s")

    # ── 5. Save JSON ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "train_date":   args.train,
            "test_date":    args.test,
            "lambda":       float("inf") if np.isinf(lam) else lam,
            "weights":      "vega",
            "n_factors":    N, "r_n": R_N,
            "M_paths":      args.M_paths,
            "seed":         args.seed,
            "max_maturities": args.max_maturities,
            "max_strikes":    args.max_strikes,
            "note": ("IS = all maturities of train_date. "
                     "Temporal OOS = same (H,rho,nu/eta) + xi0 from test_date.")
        },
        "results": results,
    }
    _save(out_dir / "temporal_oos_results.json", output)

    # ── 6. Summary table ──────────────────────────────────────────────────────
    header = (f"\nTemporal OOS: train={args.train}  test={args.test}  lam={lam}\n"
              f"{'Model':<10} {'H':>6} {'rho':>7} {'nu/eta':>7} "
              f"{'iv_IS':>8} {'sk_IS':>8} {'iv_tOOS':>9} {'sk_tOOS':>9}\n"
              + "-" * 72)
    print(header)
    lines = [header]

    for name, rec in results.items():
        p   = rec["params"]
        is_ = rec["in_sample"]
        oos = rec["temporal_oos"]
        nu_eta = p.get("nu", p.get("eta", float("nan")))
        line = (f"{name:<10} {p['H']:>6.3f} {p['rho']:>7.3f} {nu_eta:>7.3f} "
                f"{is_['iv_rmse']:>8.4f} {is_['skew_rmse']:>8.4f} "
                f"{oos['iv_rmse']:>9.4f} {oos['skew_rmse']:>9.4f}")
        print(line)
        lines.append(line)

    with open(out_dir / "summary_table.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
