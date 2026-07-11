"""Diagnostic: which aB slices produce a large ATM skew via SVI, and why?

Mirrors _reprice_ab() in plot_07b_combined.py exactly: same params, same
M_paths, same seed. Then runs SVI per slice and prints parameters + skew
derivative for every T, flagging spikes.

Usage
-----
    python scripts/_diag_svi_ab.py
"""
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
             / "2024-08-05_2024-08-06_42_lam0.0/temporal_oos_results.json")
M_PATHS   = 100_000
SEED      = 52          # args.seed=42, train seed = args.seed + 10

SPIKE_THRESHOLD = 1.0   # |dσ/dk|_{k=0}| > this gets flagged


def main():
    print(f"Loading {DATA_CSV.name} ...")
    surf    = load_spx_csv(DATA_CSV)
    xi0_m, xi0_v = fit_xi0_from_surface(surf)
    fv      = PiecewiseConstantForwardVariance(xi0_m, xi0_v)
    S0      = float(surf.forwards.mean())
    K_per_T = surf.strikes_per_T()
    F_per_T = surf.forward_per_T()

    with open(JSON_PATH) as f:
        data = json.load(f)

    for name, kernel in [("aB-L2", "L2"), ("aB-geo", "geo")]:
        p      = data["results"][name]["params"]
        params = _build_ab_params(p["H"], 20, p["eta"], p["rho"], kernel, 2.5)

        print(f"\n{'='*72}")
        print(f"Model: {name}  H={p['H']:.4f}  eta={p['eta']:.4f}  rho={p['rho']:.4f}")
        print(f"{'='*72}")

        print("Repricing (M_paths=100k) ...")
        ivs_all, _ = abergomi_iv_surface(
            params, fv, S0, K_per_T,
            M_paths=M_PATHS, qmc=False, seed=SEED,
        )

        print(f"\n{'T':>8}  {'n_fin':>6}  {'n_atm':>6}  "
              f"{'b':>7}  {'rho_s':>7}  {'m':>7}  {'s':>7}  "
              f"{'σ_ATM':>7}  {'dσ/dk':>8}  note")
        print("-" * 90)

        spikes = []
        for T in sorted(ivs_all):
            iv  = np.asarray(ivs_all[T])
            K   = K_per_T[T]
            F   = F_per_T[T]
            k   = np.log(K / F)
            fin = np.isfinite(iv) & (iv > 0)
            k_c, iv_c = k[fin], iv[fin]
            n_atm = int(np.sum(np.abs(k_c) < 0.15))

            if len(k_c) < 5:
                print(f"{T:8.4f}  {len(k_c):6d}  {n_atm:6d}  — skip (too few)")
                continue

            try:
                fit = fit_svi_slice(k_c, iv_c, T)
            except Exception as e:
                print(f"{T:8.4f}  {len(k_c):6d}  {n_atm:6d}  fit failed: {e}")
                continue

            skew     = fit.atm_skew_derivative()
            sigma_atm = fit.atm_iv()
            note = ""
            if abs(skew) > SPIKE_THRESHOLD:
                note = f"*** SPIKE  w'(0)={fit.b*(fit.rho - fit.m/np.sqrt(fit.m**2+fit.s**2)):.4f}"
                spikes.append(T)

            print(f"{T:8.4f}  {len(k_c):6d}  {n_atm:6d}  "
                  f"{fit.b:7.4f}  {fit.rho:7.4f}  {fit.m:7.4f}  {fit.s:7.4f}  "
                  f"{sigma_atm:7.4f}  {skew:8.4f}  {note}")

        if spikes:
            print(f"\n  Spike maturities: {[f'{t:.4f}' for t in spikes]}")
        else:
            print("\n  No spikes detected.")


if __name__ == "__main__":
    main()
