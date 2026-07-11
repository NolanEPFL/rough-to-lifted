"""scripts/06_lambda_sweep.py — Phase 5.  Lambda sweep for IV-RMSE vs ATM-skew.

Runs all four models × five λ_skew values on the SPX 2024-01-19 surface and
saves a single reloadable JSON for notebook analysis.

Composite loss:
    L(λ) = L_wIV  +  λ · L_skew     0 ≤ λ < ∞
    L(∞) = L_skew                    pure skew calibration

CLI usage
---------
    python scripts/06_lambda_sweep.py                         # production
    python scripts/06_lambda_sweep.py --de-maxiter 5 \\
        --M-paths 10000                                       # smoke test

Results are written to results/06_lambda_sweep/<date>_<seed>/ incrementally
(JSON flushed after each (model, λ) pair so a crash loses at most one run).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.calibration.loss import IVSurfaceLoss
from src.calibration.optimizer import (
    calibrate_lifted_heston,
    calibrate_abergomi,
    LH_BOUNDS,
    AB_BOUNDS,
)
from src.lifted_heston.params import LiftedHestonParams
from src.lifted_heston.pricing import lifted_heston_iv_surface
from src.abergomi.kernel_fit import ABergomiParams
from src.abergomi.pricing import abergomi_iv_surface


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thin_surface(
    market_ivs: dict, strikes_per: dict, forwards_per: dict,
    max_mats: int = 8, max_strikes: int = 15,
) -> tuple[dict, dict, dict]:
    """Log-spaced maturity thinning + even strike thinning (matches 04_spx_comparison)."""
    all_T = np.array(sorted(market_ivs.keys()))
    if len(all_T) > max_mats:
        targets = np.exp(np.linspace(np.log(all_T[0]), np.log(all_T[-1]), max_mats))
        idx = [int(np.argmin(np.abs(all_T - t))) for t in targets]
        all_T = all_T[sorted(set(idx))]

    new_iv, new_K, new_F = {}, {}, {}
    for T in all_T:
        ivs = market_ivs[T]
        Ks  = strikes_per[T]
        if len(Ks) > max_strikes:
            idx = np.round(np.linspace(0, len(Ks) - 1, max_strikes)).astype(int)
            Ks  = Ks[idx]; ivs = ivs[idx]
        new_iv[float(T)] = ivs
        new_K[float(T)]  = Ks
        new_F[float(T)]  = forwards_per.get(float(T), 0.0)
    return new_iv, new_K, new_F


def _lambda_key(lam: float) -> str:
    return "lambda_inf" if math.isinf(lam) else f"lambda_{lam}"


def _parse_lambdas(raw: list[str]) -> list[float]:
    result = []
    for s in raw:
        if s.lower() in ("inf", "infinity", "np.inf"):
            result.append(np.inf)
        else:
            result.append(float(s))
    return result


def _run_dir(out_root: Path, date: str, seed: int) -> Path:
    d = out_root / f"{date}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ── Model runner ──────────────────────────────────────────────────────────────

def _run_lh(
    name: str, kernel: str, lam: float,
    loss_template: IVSurfaceLoss, forward_variance, S0: float,
    strikes_per: dict, common_lh: dict,
) -> tuple[dict, dict]:
    """Calibrate one Lifted Heston variant and return (record, ivs)."""
    loss = copy.copy(loss_template)
    loss.skew_lambda = lam

    res = calibrate_lifted_heston(loss, forward_variance, kernel=kernel, **common_lh)
    p = res.params
    params_obj = LiftedHestonParams(
        H=p["H"], n=common_lh["n"], r_n=common_lh["r_n"],
        nu=p["nu"], rho=p["rho"], kernel=kernel,
    )
    ivs = lifted_heston_iv_surface(params_obj, forward_variance, S0, strikes_per)

    is_comp  = loss_template.components(ivs, oos=False)
    oos_comp = loss_template.components(ivs, oos=True)

    record = {
        "lambda": float("inf") if math.isinf(lam) else lam,
        "params": res.params,
        "in_sample":    {**is_comp,  "total_loss": res.loss_in_sample},
        "out_of_sample": oos_comp,
        "n_evals":   res.n_evals,
        "elapsed_s": res.elapsed_seconds,
    }
    return record, ivs


def _run_ab(
    name: str, kernel: str, lam: float,
    loss_template: IVSurfaceLoss, forward_variance, S0: float,
    strikes_per: dict, common_ab: dict,
    seed_offset: int,
) -> tuple[dict, dict]:
    """Calibrate one aBergomi variant and return (record, ivs)."""
    loss = copy.copy(loss_template)
    loss.skew_lambda = lam

    res = calibrate_abergomi(loss, forward_variance, kernel=kernel, **common_ab)
    p = res.params

    from src.calibration.optimizer import _build_ab_params
    params_obj = _build_ab_params(
        p["H"], common_ab["n"], p["eta"], p["rho"], kernel, common_ab["r_n"]
    )
    ivs, _ = abergomi_iv_surface(
        params_obj, forward_variance, S0, strikes_per,
        M_paths=common_ab["M_paths"], qmc=False,
        seed=common_ab["seed"] + seed_offset,
    )

    is_comp  = loss_template.components(ivs, oos=False)
    oos_comp = loss_template.components(ivs, oos=True)

    record = {
        "lambda": float("inf") if math.isinf(lam) else lam,
        "params": res.params,
        "in_sample":    {**is_comp,  "total_loss": res.loss_in_sample},
        "out_of_sample": oos_comp,
        "n_evals":   res.n_evals,
        "elapsed_s": res.elapsed_seconds,
    }
    return record, ivs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda sweep: IV-RMSE vs ATM-skew")
    parser.add_argument("--date",       default="2024-01-19")
    parser.add_argument("--out",        type=Path, default=Path("results/06_lambda_sweep"))
    parser.add_argument("--lambdas",    nargs="+", default=["0", "0.01", "0.1", "1.0", "inf"])
    parser.add_argument("--n-factors",  type=int,   default=20)
    parser.add_argument("--r-n",        type=float, default=2.5)
    parser.add_argument("--M-paths",    type=int,   default=50_000)
    parser.add_argument("--de-maxiter", type=int,   default=30)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--no-lh-l2",      action="store_true", help="Skip LH-L2")
    parser.add_argument("--max-maturities", type=int,   default=None,
                        help="Thin to N log-spaced maturities (default: use all).")
    parser.add_argument("--max-strikes",    type=int,   default=None,
                        help="Thin to N strikes per maturity (default: use all).")
    args = parser.parse_args()

    lambdas = _parse_lambdas(args.lambdas)
    run_dir = _run_dir(args.out, args.date, args.seed)

    # ── 1. Load surface ───────────────────────────────────────────────────────
    csv_path = _ROOT / "data" / f"spx_{args.date}.csv"
    print(f"Loading {csv_path}...")
    surface = load_spx_csv(csv_path)
    xi0_mats, xi0_vals = fit_xi0_from_surface(surface)
    forward_variance = PiecewiseConstantForwardVariance(xi0_mats, xi0_vals)
    S0 = float(surface.forwards.mean())

    market_ivs   = surface.ivs_per_T()
    strikes_per  = surface.strikes_per_T()
    forwards_per = surface.forward_per_T()

    # Optional thinning (for smoke tests; omit for production)
    if args.max_maturities or args.max_strikes:
        market_ivs, strikes_per, forwards_per = _thin_surface(
            market_ivs, strikes_per, forwards_per,
            max_mats=args.max_maturities or len(market_ivs),
            max_strikes=args.max_strikes or 9999,
        )

    print(f"Surface: {len(market_ivs)} maturities, "
          f"{sum(len(v) for v in market_ivs.values())} points, S0={S0:.2f}")

    # ── 2. Loss templates ─────────────────────────────────────────────────────
    loss_template = IVSurfaceLoss(
        market_ivs=market_ivs,
        strikes=strikes_per,
        S0=S0,
        forwards=forwards_per,
        weights="vega",
        skew_lambda=0.0,          # overridden per (model, λ) run
        in_sample_T_max=1.0,
    )

    # Minimal 5-strike grid per maturity for lambda=inf (pure skew).
    # Only need to bracket 0.98*F_T and 1.02*F_T for the 98/102 interpolation.
    # 5 strikes x n_maturities vs 6936 points -> ~30x faster for LH at lam=inf.
    _skew_strikes = {
        T: np.array([F * 0.96, F * 0.98, F * 1.00, F * 1.02, F * 1.04])
        for T, F in forwards_per.items() if T in market_ivs
    }
    _skew_ivs = {}
    for T, K_skew in _skew_strikes.items():
        K_mkt = strikes_per[T]
        iv_mkt = market_ivs[T]
        k_mkt  = np.log(K_mkt / forwards_per[T])
        k_skew = np.log(K_skew / forwards_per[T])
        order  = np.argsort(k_mkt)
        _skew_ivs[T] = np.interp(k_skew, k_mkt[order], iv_mkt[order])

    loss_skew_only = IVSurfaceLoss(
        market_ivs=_skew_ivs,
        strikes=_skew_strikes,
        S0=S0,
        forwards=forwards_per,
        weights="equal",           # vega undefined at synthetic 5-point grid
        skew_lambda=np.inf,
        in_sample_T_max=1.0,
    )

    # ── 3. Common optimizer kwargs ────────────────────────────────────────────
    common_lh = dict(
        n=args.n_factors, r_n=args.r_n, seed=args.seed,
        de_popsize=10, de_maxiter=args.de_maxiter, refine=True,
    )
    common_ab = dict(
        n=args.n_factors, r_n=args.r_n, seed=args.seed,
        M_paths=args.M_paths, de_popsize=10, de_maxiter=args.de_maxiter,
        refine=True,
        final_M_paths=min(args.M_paths * 4, 100_000),
    )

    # ── 4. Define model list ──────────────────────────────────────────────────
    models = [
        ("LH-geo",  "lh", "geometric"),
        ("aB-L2",   "ab", "l2"),
        ("aB-geo",  "ab", "geometric"),
    ]
    if not args.no_lh_l2:
        models.insert(1, ("LH-L2", "lh", "l2"))

    # ── 5. Metadata ───────────────────────────────────────────────────────────
    lam_meta = ["inf" if math.isinf(l) else l for l in lambdas]
    metadata = {
        "date":           args.date,
        "lambdas":        lam_meta,
        "weights":        "vega",
        "in_sample_T_max": 1.0,
        "n_factors":      args.n_factors,
        "r_n":            args.r_n,
        "M_paths":        args.M_paths,
        "seed":           args.seed,
    }
    _save_json(run_dir / "metadata.json", metadata)

    # ── 6. Sweep ──────────────────────────────────────────────────────────────
    sweep: dict = {"metadata": metadata, "results": {}}

    for model_name, family, kernel in models:
        sweep["results"][model_name] = {}
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (kernel={kernel})")
        print(f"{'='*60}")

        for lam in lambdas:
            lkey = _lambda_key(lam)
            lam_str = "inf" if math.isinf(lam) else str(lam)
            print(f"\n  lam={lam_str} ...", flush=True)
            t0 = time.perf_counter()

            try:
                # For lambda=inf use the minimal 5-strike skew grid (~30x faster)
                active_loss = loss_skew_only if math.isinf(lam) else loss_template
                active_K    = _skew_strikes  if math.isinf(lam) else strikes_per

                if family == "lh":
                    record, _ = _run_lh(
                        model_name, kernel, lam,
                        active_loss, forward_variance, S0, active_K, common_lh,
                    )
                else:
                    seed_off = {"LH-geo": 0, "LH-L2": 1, "aB-L2": 2, "aB-geo": 3}
                    record, _ = _run_ab(
                        model_name, kernel, lam,
                        active_loss, forward_variance, S0, active_K, common_ab,
                        seed_offset=seed_off.get(model_name, 0),
                    )
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                record = {"lambda": lam_str, "error": str(exc)}

            elapsed = time.perf_counter() - t0
            sweep["results"][model_name][lkey] = record

            # Print one-line summary
            p = record.get("params", {})
            is_ = record.get("in_sample", {})
            oos = record.get("out_of_sample", {})
            print(
                f"  {model_name} lam={lam_str:>5}  "
                f"H={p.get('H', float('nan')):.3f}  "
                f"rho={p.get('rho', float('nan')):.3f}  "
                f"iv_IS={is_.get('iv_rmse', float('nan')):.4f}  "
                f"sk_IS={is_.get('skew_rmse', float('nan')):.4f}  "
                f"iv_OOS={oos.get('iv_rmse', float('nan')):.4f}  "
                f"t={elapsed:.0f}s",
                flush=True,
            )

            # Incremental save
            _save_json(run_dir / "sweep_results.json", sweep)

    print(f"\nAll done. Results in {run_dir}")


if __name__ == "__main__":
    main()
