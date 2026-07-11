"""Calibration pipeline: differential evolution + L-BFGS-B refinement.

CRN strategy for aBergomi: passing the same ``seed`` to every
``abergomi_iv_surface`` call throughout the DE + L-BFGS search makes the
loss surface deterministic (same noise every evaluation).  Final reported
standard errors use a fresh, larger independent sample (final_M_paths).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import differential_evolution, minimize

from ..lifted_heston.params import LiftedHestonParams, geometric_grid
from ..lifted_heston.pricing import lifted_heston_iv_surface
from ..abergomi.kernel_fit import ABergomiParams, abergomi_l2_kernel
from ..abergomi.pricing import abergomi_iv_surface
from ..common.forward_variance import ForwardVariance
from .loss import IVSurfaceLoss

# Parameter bounds from CONTEXT.md §6.2  (nu/eta, rho, H)
LH_BOUNDS = [(0.05, 3.0), (-0.99, 0.0), (0.02, 0.49)]
AB_BOUNDS  = [(0.5,  5.0), (-0.99, 0.0), (0.02, 0.49)]


@dataclass
class CalibrationResult:
    params:              dict
    loss_in_sample:      float
    loss_out_of_sample:  float | None
    n_evals:             int
    elapsed_seconds:     float
    de_history:          list = field(default_factory=list)
    refined_history:     list = field(default_factory=list)


# ── Kernel cache for aBergomi (expensive L² fit) ───────────────────────────
_ab_kernel_cache: dict = {}


def _build_ab_params(
    H: float, n: int, eta: float, rho: float, kernel: str, r_n: float
) -> ABergomiParams:
    """Construct ABergomiParams with kernel cached by rounded H.

    Avoids refitting the kernel for every DE evaluation that merely changes
    eta/rho while keeping H approximately constant.
    """
    H_key = round(H, 4)
    cache_key = (H_key, n, kernel, r_n)

    if cache_key not in _ab_kernel_cache:
        if kernel == "l2":
            c, x, _ = abergomi_l2_kernel(H_key, n, seed=42)
        else:
            c, x = geometric_grid(H_key, n, r_n)
        _ab_kernel_cache[cache_key] = (c.copy(), x.copy())

    c, x = _ab_kernel_cache[cache_key]

    # Bypass __post_init__ kernel-fitting; set fields manually
    params = object.__new__(ABergomiParams)
    params.H      = H
    params.n      = n
    params.eta    = eta
    params.rho    = rho
    params.kernel = kernel
    params.eps    = 1.0 / 365.0
    params.T_max  = 2.0
    params.r_n    = r_n
    params.c      = c
    params.x      = x
    return params


def calibrate_lifted_heston(
    loss: IVSurfaceLoss,
    forward_variance: ForwardVariance,
    n: int = 20,
    r_n: float = 2.5,
    kernel: str = "geometric",
    seed: int = 42,
    de_popsize: int = 10,
    de_maxiter: int = 30,
    refine: bool = True,
) -> CalibrationResult:
    """Calibrate (nu, rho, H) for lifted Heston.

    kernel : 'geometric' (default) or 'l2' (L²-fit kernel, Phase 4 extension).
    Uses DE for global search then L-BFGS-B refinement.
    """
    S0          = loss.S0
    strikes_per = loss.strikes
    evals       = [0]
    de_hist: list[float] = []

    # Kernel cache for LH-L² (same H → same (c, x), expensive to refit)
    _lh_kernel_cache: dict = {}

    def _make_params(nu, rho, H):
        if kernel == "geometric":
            return LiftedHestonParams(H=H, n=n, r_n=r_n, nu=nu, rho=rho,
                                      kernel="geometric")
        # LH-L²: reuse cached kernel for the same rounded H
        H_key = round(H, 4)
        if H_key not in _lh_kernel_cache:
            from ..abergomi.kernel_fit import abergomi_l2_kernel
            c, x, _ = abergomi_l2_kernel(H_key, n, seed=42)
            _lh_kernel_cache[H_key] = (c, x)
        c, x = _lh_kernel_cache[H_key]
        params = object.__new__(LiftedHestonParams)
        params.H = H; params.n = n; params.r_n = r_n
        params.nu = nu; params.rho = rho; params.kernel = kernel
        params.c = c; params.x = x
        return params

    def obj(theta: np.ndarray) -> float:
        nu, rho, H = theta
        evals[0] += 1
        try:
            params = _make_params(nu, rho, H)
            ivs    = lifted_heston_iv_surface(params, forward_variance, S0, strikes_per)
            v      = loss(ivs)
        except Exception:
            v = 1e6
        de_hist.append(v)
        return v

    t0 = time.perf_counter()
    de_res = differential_evolution(
        obj, LH_BOUNDS,
        popsize=de_popsize, maxiter=de_maxiter,
        seed=seed, polish=False, workers=1, tol=1e-4,
    )
    best = de_res.x

    ref_hist: list[float] = []
    if refine:
        def obj_ref(theta: np.ndarray) -> float:
            v = obj(theta)
            ref_hist.append(v)
            return v
        opt = minimize(
            obj_ref, x0=best, method="L-BFGS-B", bounds=LH_BOUNDS,
            options={"maxiter": 100, "eps": 1e-4, "ftol": 1e-12},
        )
        if opt.fun < de_res.fun:
            best = opt.x

    nu, rho, H = best
    params_best = LiftedHestonParams(H=H, n=n, r_n=r_n, nu=nu, rho=rho)
    ivs_best    = lifted_heston_iv_surface(params_best, forward_variance, S0, strikes_per)
    loss_is     = loss(ivs_best)
    loss_oos    = loss.out_of_sample(ivs_best)

    return CalibrationResult(
        params={"nu": float(nu), "rho": float(rho), "H": float(H)},
        loss_in_sample=loss_is,
        loss_out_of_sample=loss_oos,
        n_evals=evals[0],
        elapsed_seconds=time.perf_counter() - t0,
        de_history=de_hist,
        refined_history=ref_hist,
    )


def calibrate_abergomi(
    loss: IVSurfaceLoss,
    forward_variance: ForwardVariance,
    n: int = 20,
    kernel: str = "l2",
    r_n: float = 2.5,
    M_paths: int = 50_000,
    n_steps_per_year: int = 100,
    seed: int = 42,
    de_popsize: int = 10,
    de_maxiter: int = 30,
    refine: bool = True,
    final_M_paths: int = 200_000,
) -> CalibrationResult:
    """Calibrate (eta, rho, H) for aBergomi.

    CRN: ``seed`` is fixed for all evaluations during DE+L-BFGS, making the
    loss surface deterministic (same random numbers every call).  The final
    standard errors are computed with a fresh, larger sample.
    """
    S0          = loss.S0
    strikes_per = loss.strikes
    evals       = [0]
    de_hist: list[float] = []

    def obj(theta: np.ndarray) -> float:
        eta, rho, H = theta
        evals[0] += 1
        try:
            params = _build_ab_params(H, n, eta, rho, kernel, r_n)
            # CRN: same seed → same noise → smooth loss surface
            ivs, _ = abergomi_iv_surface(
                params, forward_variance, S0, strikes_per,
                M_paths=M_paths, n_steps_per_year=n_steps_per_year,
                qmc=False, seed=seed,
            )
            v = loss(ivs)
        except Exception:
            v = 1e6
        de_hist.append(v)
        return v

    t0 = time.perf_counter()
    de_res = differential_evolution(
        obj, AB_BOUNDS,
        popsize=de_popsize, maxiter=de_maxiter,
        seed=seed, polish=False, workers=1, tol=1e-4,
    )
    best = de_res.x

    ref_hist: list[float] = []
    if refine:
        def obj_ref(theta: np.ndarray) -> float:
            v = obj(theta)
            ref_hist.append(v)
            return v
        opt = minimize(
            obj_ref, x0=best, method="L-BFGS-B", bounds=AB_BOUNDS,
            options={"maxiter": 100, "eps": 1e-4, "ftol": 1e-12},
        )
        if opt.fun < de_res.fun:
            best = opt.x

    eta, rho, H = best
    params_best = _build_ab_params(H, n, eta, rho, kernel, r_n)

    # In-sample + OOS via frozen-CRN surface
    ivs_crn, _ = abergomi_iv_surface(
        params_best, forward_variance, S0, strikes_per,
        M_paths=M_paths, n_steps_per_year=n_steps_per_year,
        qmc=False, seed=seed,
    )
    loss_is  = loss(ivs_crn)
    loss_oos = loss.out_of_sample(ivs_crn)

    # Final standard errors with fresh large sample (CONTEXT.md §6.3)
    ivs_final, _ = abergomi_iv_surface(
        params_best, forward_variance, S0, strikes_per,
        M_paths=final_M_paths, n_steps_per_year=n_steps_per_year,
        qmc=False, seed=seed + 999,
    )
    # Recompute loss for the final sample (more accurate than CRN estimate)
    loss_is_final  = loss(ivs_final)
    loss_oos_final = loss.out_of_sample(ivs_final)

    return CalibrationResult(
        params={"eta": float(eta), "rho": float(rho), "H": float(H)},
        loss_in_sample=loss_is_final,
        loss_out_of_sample=loss_oos_final,
        n_evals=evals[0],
        elapsed_seconds=time.perf_counter() - t0,
        de_history=de_hist,
        refined_history=ref_hist,
    )
