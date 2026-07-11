"""scripts/cboe_2018_five_models_laminf.py - single-model lambda=inf calibration.

Calibrates ONE of the five Ch.8 models (LH-geo, LH-L2, aB-geo, aB-L2, TF-LH) at
lambda=inf (pure ATM-skew RMSE) on the 2018-06-20 SPX surface. Uses the same
machinery and grid conventions as scripts/07_temporal_oos.py and
scripts/07b_temporal_oos_two_factor.py: thin 8x15 + 5-strike skew grid at
{0.96, 0.98, 1.00, 1.02, 1.04} * F, weights="equal", in_sample_T_max=None,
seed=42.

Designed to be launched in parallel for the five models (one process each).
Each invocation writes:
    results/cboe_2018_validation/five_models_laminf_<model>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import json
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss, _atm_skew_98_102
from src.calibration.optimizer import (
    calibrate_lifted_heston, calibrate_abergomi, _build_ab_params,
)
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.pricing import abergomi_iv_surface


SURFACE_CSV = Path(_ROOT) / "data" / "spx_2018-06-20.csv"
OUT_DIR = Path(_ROOT) / "results" / "cboe_2018_validation"
S0_SPOT = 2769.59
MAX_MATURITIES = 8
MAX_STRIKES = 15
SKEW_MULTS = np.array([0.96, 0.98, 1.00, 1.02, 1.04])
SEED = 42


def load_tf_calibrator():
    """Load calibrate_two_factor_lh from scripts/07b_temporal_oos_two_factor.py
    via importlib (the filename starts with a digit and can't be import-ed)."""
    spec = importlib.util.spec_from_file_location(
        "tf_oos_module", Path(_ROOT) / "scripts" / "07b_temporal_oos_two_factor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calibrate_two_factor_lh


def _thin(mkt_full, K_full, F_full):
    all_T = np.array(sorted(mkt_full.keys()))
    if len(all_T) > MAX_MATURITIES:
        targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), MAX_MATURITIES))
        idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
        all_T = all_T[sorted(set(idx))]
    mkt, K, F = {}, {}, {}
    for T in all_T:
        ivs = mkt_full[T]; Ks = K_full[T]
        if len(Ks) > MAX_STRIKES:
            idx = np.round(np.linspace(0, len(Ks) - 1, MAX_STRIKES)).astype(int)
            Ks = Ks[idx]; ivs = ivs[idx]
        mkt[float(T)] = ivs
        K[float(T)] = Ks
        F[float(T)] = F_full[float(T)]
    return mkt, K, F


def build_skew_loss(surf):
    """Replicate the lambda=inf loss-construction logic from
    scripts/07_temporal_oos.py lines 173-189 (5-strike skew grid)."""
    mkt_full = surf.ivs_per_T()
    K_full = surf.strikes_per_T()
    F_full = surf.forward_per_T()
    mkt_thin, K_thin, F_thin = _thin(mkt_full, K_full, F_full)

    calib_K = {T: F * SKEW_MULTS for T, F in F_thin.items() if T in mkt_thin}
    calib_mkt = {}
    for T, K5 in calib_K.items():
        K_T = np.asarray(K_thin[T], dtype=float)
        iv_T = np.asarray(mkt_thin[T], dtype=float)
        F_T = float(F_thin[T])
        k_full = np.log(K_T / F_T)
        k5 = np.log(K5 / F_T)
        order = np.argsort(k_full)
        calib_mkt[T] = np.interp(k5, k_full[order], iv_T[order])

    S0_calib = float(surf.forwards.mean())   # convention from 07_temporal_oos.py
    loss = IVSurfaceLoss(
        market_ivs=calib_mkt, strikes=calib_K, S0=S0_calib, forwards=F_thin,
        weights="equal", skew_lambda=np.inf, in_sample_T_max=None,
    )
    return loss, S0_calib, calib_mkt, calib_K, F_thin


def compute_per_maturity_skew(model_name, params_obj_or_dict, fv, surf, S0_calib):
    """Evaluate the calibrated model on the FULL cleaned-surface strikes
    and return the per-maturity (empirical, model) 98/102 skew table under
    SPOT-moneyness (paper convention, matches A.3 figure)."""
    K_full = surf.strikes_per_T()
    mkt_full = surf.ivs_per_T()

    if model_name in ("lh-geo", "lh-l2"):
        kernel = "geometric" if model_name == "lh-geo" else "l2"
        p = LiftedHestonParams(
            H=params_obj_or_dict["H"], n=20, r_n=2.5, kernel=kernel,
            nu=params_obj_or_dict["nu"], rho=params_obj_or_dict["rho"],
        )
        iv_model = lifted_heston_iv_surface(p, fv, S0_calib, K_full)
    elif model_name in ("ab-geo", "ab-l2"):
        kernel = "geometric" if model_name == "ab-geo" else "l2"
        p = _build_ab_params(
            params_obj_or_dict["H"], 20, params_obj_or_dict["eta"],
            params_obj_or_dict["rho"], kernel, 2.5,
        )
        iv_model, _ = abergomi_iv_surface(
            p, fv, S0_calib, K_full, M_paths=200_000, qmc=False,
            seed=SEED + 999,
        )
    elif model_name == "tf-lh":
        from src.two_factor_lifted_heston.pricing import two_factor_lh_iv_surface
        iv_model = two_factor_lh_iv_surface(params_obj_or_dict, fv, S0_calib, K_full)
    else:
        raise ValueError(model_name)

    rows = []
    for T in sorted(K_full.keys()):
        K_T = np.asarray(K_full[T], dtype=float)
        ivs_mkt = np.asarray(mkt_full[T], dtype=float)
        ivs_mdl = np.asarray(iv_model[T], dtype=float)
        s_emp = _atm_skew_98_102(S0_SPOT, K_T, ivs_mkt)
        s_mdl = _atm_skew_98_102(S0_SPOT, K_T, ivs_mdl)
        n_nan = int((~np.isfinite(ivs_mdl)).sum())
        rows.append({
            "T": float(T),
            "empirical_skew_spot": (None if s_emp is None else float(s_emp)),
            "model_skew_spot": (None if s_mdl is None else float(s_mdl)),
            "n_strikes": int(len(K_T)),
            "n_model_iv_nan": n_nan,
        })
    return rows


def run_one(model_name: str) -> dict:
    print(f"[{model_name}] loading surface from {SURFACE_CSV}")
    surf = load_spx_csv(SURFACE_CSV, weight_scheme="equal")
    loss, S0_calib, calib_mkt, calib_K, F_thin = build_skew_loss(surf)
    n_skew_points = int(sum(len(v) for v in calib_mkt.values()))
    print(f"[{model_name}] thin grid: {len(calib_K)} maturities, 5 strikes each "
          f"= {n_skew_points} skew points; S0_calib (for vega weights) = {S0_calib:.4f}")

    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)

    t0 = time.time()
    params_for_eval = None
    loss_at_optimum = None
    n_evals = None
    elapsed_calibration = None
    convergence_note = "ok"

    if model_name == "lh-geo":
        res = calibrate_lifted_heston(loss, fv, n=20, r_n=2.5, kernel="geometric", seed=SEED)
        params_for_eval = res.params
        loss_at_optimum = float(res.loss_in_sample)
        n_evals = int(res.n_evals); elapsed_calibration = float(res.elapsed_seconds)
        calibrated_params = {"H": float(res.params["H"]), "nu": float(res.params["nu"]),
                              "rho": float(res.params["rho"])}
    elif model_name == "lh-l2":
        res = calibrate_lifted_heston(loss, fv, n=20, r_n=2.5, kernel="l2", seed=SEED)
        params_for_eval = res.params
        loss_at_optimum = float(res.loss_in_sample)
        n_evals = int(res.n_evals); elapsed_calibration = float(res.elapsed_seconds)
        calibrated_params = {"H": float(res.params["H"]), "nu": float(res.params["nu"]),
                              "rho": float(res.params["rho"])}
    elif model_name == "ab-geo":
        res = calibrate_abergomi(loss, fv, n=20, r_n=2.5, kernel="geometric",
                                  M_paths=50_000, seed=SEED, final_M_paths=200_000)
        params_for_eval = res.params
        loss_at_optimum = float(res.loss_in_sample)
        n_evals = int(res.n_evals); elapsed_calibration = float(res.elapsed_seconds)
        calibrated_params = {"H": float(res.params["H"]), "eta": float(res.params["eta"]),
                              "rho": float(res.params["rho"])}
    elif model_name == "ab-l2":
        res = calibrate_abergomi(loss, fv, n=20, r_n=2.5, kernel="l2",
                                  M_paths=50_000, seed=SEED, final_M_paths=200_000)
        params_for_eval = res.params
        loss_at_optimum = float(res.loss_in_sample)
        n_evals = int(res.n_evals); elapsed_calibration = float(res.elapsed_seconds)
        calibrated_params = {"H": float(res.params["H"]), "eta": float(res.params["eta"]),
                              "rho": float(res.params["rho"])}
    elif model_name == "tf-lh":
        calibrate_tf = load_tf_calibrator()
        p_dict, p_obj, n_evals_local, elapsed_local = calibrate_tf(
            loss, fv, seed=SEED, de_popsize=10, de_maxiter=30, refine=True,
        )
        params_for_eval = p_obj
        n_evals = int(n_evals_local); elapsed_calibration = float(elapsed_local)
        # recompute the loss at optimum on the calibration grid
        from src.two_factor_lifted_heston.pricing import two_factor_lh_iv_surface
        iv_calib = two_factor_lh_iv_surface(p_obj, fv, S0_calib, calib_K)
        loss_at_optimum = float(loss(iv_calib))
        calibrated_params = {k: (float(v) if v is not None else None) for k, v in p_dict.items()}
    else:
        raise ValueError(f"unknown model: {model_name}")

    elapsed_total = time.time() - t0

    # Heuristic convergence flag
    if loss_at_optimum is None or not np.isfinite(loss_at_optimum):
        convergence_note = "non-finite loss"
    elif loss_at_optimum > 1.0:
        convergence_note = f"high final loss ({loss_at_optimum:.3f})"
    if model_name in ("ab-geo", "ab-l2") and elapsed_calibration is not None and elapsed_calibration < 60:
        convergence_note = "suspiciously fast for aBergomi MC"

    print(f"[{model_name}] elapsed total = {elapsed_total:.1f}s, calibration = "
          f"{elapsed_calibration:.1f}s, n_evals = {n_evals}, loss = {loss_at_optimum:.6f}, "
          f"convergence = {convergence_note}")
    print(f"[{model_name}] params: {json.dumps(calibrated_params, indent=2)}")

    # Per-maturity skew table on the FULL surface (spot moneyness, paper convention)
    print(f"[{model_name}] computing per-maturity model skew on full surface ...")
    skew_rows = compute_per_maturity_skew(model_name, params_for_eval, fv, surf, S0_calib)

    out = {
        "model": model_name,
        "config": {
            "n_factors": 20, "r_n": 2.5,
            "kernel": (
                "geometric" if model_name in ("lh-geo", "ab-geo") else
                "l2" if model_name in ("lh-l2", "ab-l2") else
                "n/a"
            ),
            "skew_lambda": "inf",
            "weights": "equal",
            "calib_grid": {"max_maturities": MAX_MATURITIES, "max_strikes": MAX_STRIKES,
                            "skew_mults": SKEW_MULTS.tolist()},
            "seed": SEED,
            "M_paths_aB_calib": 50_000 if model_name in ("ab-geo", "ab-l2") else None,
            "M_paths_aB_final": 200_000 if model_name in ("ab-geo", "ab-l2") else None,
        },
        "calibrated_params": calibrated_params,
        "loss_at_optimum_skew_RMSE": loss_at_optimum,
        "n_evals": n_evals,
        "elapsed_calibration_seconds": elapsed_calibration,
        "elapsed_total_seconds": float(elapsed_total),
        "convergence_note": convergence_note,
        "skew_per_maturity": skew_rows,
        "S0_spot": S0_SPOT,
        "S0_calib_loss_vega": S0_calib,
        "xi0_anchored_at_0": float(xi0_v[0]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         choices=["lh-geo", "lh-l2", "ab-geo", "ab-l2", "tf-lh"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = run_one(args.model)

    out_path = OUT_DIR / f"five_models_laminf_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda x: None if x != x else float(x)))
    print(f"[{args.model}] wrote {out_path}")


if __name__ == "__main__":
    main()
