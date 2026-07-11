"""Exact OU Monte Carlo simulator for aBergomi.

Strategy A (CONTEXT.md §3.6): simulate the joint Gaussian vector
(ΔW, G^{(1)}, …, G^{(n)}) per time step.  Covariances (time-independent):

    Var(ΔW) = Δ
    Cov(ΔW, G^{(i)}) = (1 − e^{−x_i Δ}) / x_i
    Cov(G^{(i)}, G^{(j)}) = (1 − e^{−(x_i+x_j) Δ}) / (x_i+x_j)

The (n+1)×(n+1) covariance is Cholesky-factored once.  ΔW^⊥ ~ N(0,Δ) is
independent.

QMC implementation
------------------
Correct Sobol QMC for path-dependent payoffs requires treating each PATH as a
single point in d = n_steps × (n+2) dimensions (CONTEXT.md: "generate
2 × M_paths × N_steps × (n+1) standard normals via Box-Muller from Sobol
points").  An equidistributed sequence in lower dimension does NOT give the
correct joint distribution across time steps, causing a systematic bias in
E[S_T].

To manage memory the noise array is generated in batches of at most
_QMC_MEM_CAP bytes, each batch using an independently scrambled Sobol
sequence (randomised QMC).  This preserves unbiasedness and gives within-batch
variance reduction.

For qmc=False plain numpy default_rng is used (always correct, no memory
overhead).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .kernel_fit import ABergomiParams
from ..common.forward_variance import ForwardVariance

_QMC_MEM_CAP = 300 * 1024 * 1024   # 300 MB per Sobol batch


def _build_step_covariance(x: np.ndarray, dt: float) -> np.ndarray:
    """Build the (n+1)×(n+1) one-step covariance matrix.

    Layout: index 0 = ΔW; indices 1..n = G^{(1)}, …, G^{(n)}.
    A tiny diagonal jitter is added to ensure strict positive-definiteness
    (when x_i·dt ≪ 1, Cov(ΔW, G^{(i)}) → Var(ΔW), nearly singular).
    """
    n = len(x)
    Sigma = np.empty((n + 1, n + 1))
    Sigma[0, 0] = dt

    cov_dW_G = (1.0 - np.exp(-x * dt)) / x
    Sigma[0, 1:] = cov_dW_G
    Sigma[1:, 0] = cov_dW_G

    x_sum = x[:, None] + x[None, :]
    Sigma[1:, 1:] = (1.0 - np.exp(-x_sum * dt)) / x_sum

    jitter = 1e-10 * np.trace(Sigma) / (n + 1)
    Sigma += jitter * np.eye(n + 1)
    return Sigma


def _build_y_variance_grid(
    c: np.ndarray, x: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    """Var(Y_{t_k}) = Σ_{i,j} c_i c_j (1−e^{−(x_i+x_j)t}) / (x_i+x_j) for all t_k."""
    x_sum    = x[:, None] + x[None, :]
    cc       = c[:, None] * c[None, :]
    ratios   = cc / x_sum
    A        = ratios.sum()
    exp_term = np.exp(-x_sum[:, :, None] * t_grid[None, None, :])
    var_Y    = A - (ratios[:, :, None] * exp_term).sum(axis=(0, 1))
    return np.maximum(var_Y, 0.0)


def _qmc_noise_batch(M: int, d: int, seed: int) -> np.ndarray:
    """Generate M standard-normal vectors of dimension d from Sobol(d) sequence."""
    from scipy.stats.qmc import Sobol
    from scipy.stats import norm as _norm
    sobol = Sobol(d=d, scramble=True, seed=int(seed % (2**32)))
    u     = sobol.random(M)
    u     = np.clip(u, 1e-10, 1.0 - 1e-10)
    return _norm.ppf(u)


def _generate_noise(
    M_paths: int,
    n_steps: int,
    n_plus2: int,
    qmc: bool,
    seed: int,
) -> np.ndarray:
    """Return noise array of shape (M_paths, n_steps, n_plus2).

    For qmc=True: each path is treated as a d = n_steps × n_plus2 dimensional
    point drawn from a scrambled Sobol sequence, giving correct joint
    equidistribution across all time steps.  Large requests are split into
    batches (each with a different scramble seed) to respect _QMC_MEM_CAP.

    For qmc=False: plain numpy pseudo-random.
    """
    if not qmc:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((M_paths, n_steps, n_plus2))

    # ── QMC: full-path dimension d = n_steps × n_plus2 ────────────────────
    d                  = n_steps * n_plus2
    bytes_per_point    = d * 8                               # float64
    M_batch_max        = max(1, _QMC_MEM_CAP // bytes_per_point)

    chunks = []
    batch  = 0
    start  = 0
    while start < M_paths:
        end  = min(start + M_batch_max, M_paths)
        M_b  = end - start
        # Independent scramble per batch (randomised QMC)
        batch_seed = (seed + batch * 1_000_003) % (2**32)
        Z_b  = _qmc_noise_batch(M_b, d, batch_seed)         # (M_b, d)
        chunks.append(Z_b.reshape(M_b, n_steps, n_plus2))
        start = end
        batch += 1

    return np.concatenate(chunks, axis=0)                   # (M_paths, n_steps, n_plus2)


def simulate_abergomi(
    params: ABergomiParams,
    forward_variance: ForwardVariance,
    S0: float,
    T: float,
    M_paths: int,
    n_steps: int,
    antithetic: bool = True,
    qmc: bool = True,
    seed: int = 42,
    record_at: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate aBergomi paths using the exact OU step (Strategy A).

    Parameters
    ----------
    record_at : optional sorted 1-d array of times ≤ T to record log_S.
                Times are snapped to the nearest grid point.

    Returns
    -------
    times  : 1-d array of actual recording times (snapped).
    log_S  : shape (M_eff, len(times)), M_eff = 2*M_paths if antithetic else M_paths.
    """
    c, x     = params.c, params.x
    eta, rho = params.eta, params.rho
    n        = len(c)
    n_p2     = n + 2           # correlated (ΔW, G^{(1)}, …, G^{(n)}) + independent ΔW^⊥

    dt     = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # ── One-time precomputation ───────────────────────────────────────────────
    Sigma_step  = _build_step_covariance(x, dt)
    L_step      = np.linalg.cholesky(Sigma_step)
    var_Y_grid  = _build_y_variance_grid(c, x, t_grid)
    xi0_grid    = np.asarray(forward_variance(t_grid), dtype=float)
    exp_neg_xdt = np.exp(-x * dt)
    sqrt_1mrho2 = float(np.sqrt(max(1.0 - rho**2, 0.0)))
    sqrt_dt     = float(np.sqrt(dt))

    # ── Recording ─────────────────────────────────────────────────────────────
    if record_at is None:
        rec_times_req = np.array([T])
    else:
        rec_times_req = np.sort(np.asarray(record_at, dtype=float))
    rec_idx   = np.array([int(np.argmin(np.abs(t_grid - t))) for t in rec_times_req])
    rec_times = t_grid[rec_idx]
    n_rec     = len(rec_idx)
    idx_to_col: dict[int, int] = {int(i): col for col, i in enumerate(rec_idx)}

    # ── Generate all noise upfront ────────────────────────────────────────────
    # Shape: (M_paths, n_steps, n_p2)
    Z_raw = _generate_noise(M_paths, n_steps, n_p2, qmc, seed)

    # ── Antithetic doubling ───────────────────────────────────────────────────
    if antithetic:
        Z_all = np.concatenate([Z_raw, -Z_raw], axis=0)  # (2*M_paths, n_steps, n_p2)
    else:
        Z_all = Z_raw
    M_eff = Z_all.shape[0]

    # ── Apply Cholesky to correlated dims (0..n) across ALL steps at once ────
    flat_Z   = Z_all[:, :, :n + 1].reshape(-1, n + 1)   # (M_eff*n_steps, n+1)
    flat_cor = flat_Z @ L_step.T                          # (M_eff*n_steps, n+1)
    corr     = flat_cor.reshape(M_eff, n_steps, n + 1)   # (M_eff, n_steps, n+1)
    dW_perp_all = sqrt_dt * Z_all[:, :, n + 1]           # (M_eff, n_steps)

    # ── Time loop ─────────────────────────────────────────────────────────────
    X     = np.zeros((M_eff, n))
    log_S = np.full(M_eff, float(np.log(S0)))
    log_S_record = np.empty((M_eff, n_rec))

    if 0 in idx_to_col:
        log_S_record[:, idx_to_col[0]] = log_S

    for k in range(n_steps):
        dW      = corr[:, k, 0]         # (M_eff,)
        G       = corr[:, k, 1:]        # (M_eff, n)
        dW_perp = dW_perp_all[:, k]     # (M_eff,)

        Y_k = X @ c
        V_k = float(xi0_grid[k]) * np.exp(
            eta * Y_k - 0.5 * eta**2 * float(var_Y_grid[k])
        )

        X     = X * exp_neg_xdt[None, :] + G

        sqrt_V_k = np.sqrt(np.maximum(V_k, 0.0))
        dB       = rho * dW + sqrt_1mrho2 * dW_perp
        log_S    = log_S - 0.5 * V_k * dt + sqrt_V_k * dB

        grid_idx = k + 1
        if grid_idx in idx_to_col:
            log_S_record[:, idx_to_col[grid_idx]] = log_S

    # Return the originally requested times (not snapped) so that output keys
    # always match the caller's maturity labels (avoids dict-key mismatch when
    # the nearest grid point slightly differs from the requested maturity).
    return rec_times_req, log_S_record
