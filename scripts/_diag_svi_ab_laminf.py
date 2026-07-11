"""Diagnostic: SVI ATM skew for lambda=inf aB params on 2024-08-05 IS surface."""
from __future__ import annotations
import json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from src.data.spx_loader import load_spx_csv, fit_xi0_from_surface
from src.common.forward_variance import PiecewiseConstantForwardVariance
from src.abergomi.pricing import abergomi_iv_surface
from src.calibration.optimizer import _build_ab_params
from src.data.svi import fit_svi_slice

DATA_CSV  = _ROOT / "data" / "spx_2024-08-05.csv"
JSON_PATH = (_ROOT / "results/07_temporal_oos"
             / "2024-08-05_2024-08-06_42_laminf/temporal_oos_results.json")

def main():
    surf = load_spx_csv(DATA_CSV)
    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0 = float(surf.forwards.mean())
    K_per_T = surf.strikes_per_T()
    F_per_T = surf.forward_per_T()

    with open(JSON_PATH) as f:
        d = json.load(f)

    for name, kernel in [("aB-L2", "L2"), ("aB-geo", "geo")]:
        p = d["results"][name]["params"]
        print(f"\n{name}: H={p['H']:.4f}  eta={p['eta']:.4f}  rho={p['rho']:.4f}")
        params = _build_ab_params(p["H"], 20, p["eta"], p["rho"], kernel, 2.5)
        ivs, _ = abergomi_iv_surface(
            params, fv, S0, K_per_T, M_paths=100_000, qmc=False, seed=52
        )
        print(f"  {'T':>8}  {'n_fin':>6}  {'n_atm':>6}  {'b':>7}  "
              f"{'rho_s':>7}  {'rmse_iv':>8}  {'dσ/dk':>8}")
        print("  " + "-" * 65)
        for T in sorted(ivs):
            if T < 0.25:
                continue
            iv = np.asarray(ivs[T])
            K  = K_per_T[T]
            F  = F_per_T[T]
            k  = np.log(K / F)
            fin = np.isfinite(iv) & (iv > 0)
            kc, ivc = k[fin], iv[fin]
            n_atm = int(np.sum(np.abs(kc) < 0.15))
            if len(kc) < 5:
                print(f"  {T:8.4f}  {len(kc):6d}  {n_atm:6d}  — skip")
                continue
            try:
                fit = fit_svi_slice(kc, ivc, T)
                skew = fit.atm_skew_derivative()
                flag = "  *** SPIKE" if abs(skew) > 1.0 else ""
                print(f"  {T:8.4f}  {len(kc):6d}  {n_atm:6d}  "
                      f"{fit.b:7.4f}  {fit.rho:7.4f}  {fit.rmse_iv:8.5f}  "
                      f"{skew:8.4f}{flag}")
            except Exception as e:
                print(f"  {T:8.4f}  fit failed: {e}")

if __name__ == "__main__":
    main()
