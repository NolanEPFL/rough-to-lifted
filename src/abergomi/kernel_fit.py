"""L²-optimal exponential-sum kernel fit for aBergomi (eq. 5.4 of the thesis).

We solve

    (c̃_i, x̃_i)_{i=1}^n = argmin_{c_i > 0, x_i > 0}
        ∫_eps^{T_max} (K_H(t) - Σ_i c_i exp(-x_i t))² w(t) dt

via:
1. Outer optimisation on log-speeds y_i = log(x_i) ∈ ℝ (no constraints).
2. For fixed {x_i}, weights {c_i ≥ 0} solved by NNLS on the discretised integral.
3. 5 random restarts of the L-BFGS outer solver, log-uniform initial speeds in
   [log 0.1, log 1000].

This is more stable than joint optimisation of (c_i, x_i) because (a) the inner
NNLS is convex and exact, and (b) the outer problem in y_i is smooth even though
the joint problem has a non-convex feasibility cone.

The aBergomi paper (Zhu et al. 2021) does not prescribe the weight w; we use
w(t) ≡ 1 by default. CONTEXT.md Section 3.3 covers the choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize, nnls

from ..lifted_heston.kernel import K_H, K_n, kernel_l2_error


def abergomi_l2_kernel(
    H: float,
    n: int,
    eps: float = 1.0 / 365.0,
    T_max: float = 2.0,
    n_restarts: int = 5,
    n_quad: int = 200,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """L²-optimal sum-of-exponentials approximation of K_H on [eps, T_max].

    Returns
    -------
    c : positive weights, shape (n,).
    x : positive speeds, sorted ascending, shape (n,).
    err : final L² error.
    """
    # Build log-spaced quadrature grid
    t_grid = np.logspace(np.log10(eps), np.log10(T_max), n_quad)

    # Trapezoidal quadrature weights
    w = np.empty(n_quad)
    w[0] = 0.5 * (t_grid[1] - t_grid[0])
    w[-1] = 0.5 * (t_grid[-1] - t_grid[-2])
    w[1:-1] = 0.5 * (t_grid[2:] - t_grid[:-2])
    sqrt_w = np.sqrt(w)

    # Weighted target vector
    K_H_vals = K_H(t_grid, H)
    b_w = sqrt_w * K_H_vals

    def inner_solve(y: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.exp(y)
        design = np.exp(-x_arr[None, :] * t_grid[:, None])   # (n_quad, n)
        A_w = sqrt_w[:, None] * design                        # weighted rows
        c_arr, residual = nnls(A_w, b_w)
        return float(residual), c_arr

    def obj(y: np.ndarray) -> float:
        return inner_solve(y)[0]

    bounds = [(np.log(0.01), np.log(1e4))] * n
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best_c: np.ndarray | None = None
    best_x: np.ndarray | None = None

    for _ in range(n_restarts):
        y0 = rng.uniform(np.log(0.1), np.log(1000.0), n)
        result = minimize(obj, y0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-15})
        if result.fun < best_err:
            best_err = result.fun
            best_x = np.exp(result.x)
            _, best_c = inner_solve(result.x)

    idx = np.argsort(best_x)
    c_out = best_c[idx]
    x_out = best_x[idx]

    assert np.all(c_out >= 0), "NNLS returned negative c_i"
    assert np.all(x_out > 0), "Some x_i ≤ 0"

    # Replace any zero weights from NNLS with a tiny positive value
    c_out = np.maximum(c_out, 1e-300)

    return c_out, x_out, best_err


@dataclass
class ABergomiParams:
    """Bundle of aBergomi parameters.

    Attributes
    ----------
    H : Hurst index (used only to fit the kernel).
    n : number of factors.
    kernel : either 'l2' (default) or 'geometric' (uses lifted_heston.params grid).
    eta : vol-of-vol > 0.
    rho : leverage in [-1, 1].
    eps, T_max : kernel-fit interval (only used if kernel == 'l2').
    r_n : geometric ratio (only used if kernel == 'geometric').
    c, x : derived weights and speeds.
    """
    H: float
    n: int
    eta: float
    rho: float
    kernel: str = "l2"
    eps: float = 1.0 / 365.0
    T_max: float = 2.0
    r_n: float = 2.5
    c: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.eta <= 0:
            raise ValueError(f"eta must be > 0, got {self.eta}")
        if self.kernel == "l2":
            self.c, self.x, _err = abergomi_l2_kernel(
                self.H, self.n, eps=self.eps, T_max=self.T_max
            )
        elif self.kernel == "geometric":
            from ..lifted_heston.params import geometric_grid
            self.c, self.x = geometric_grid(self.H, self.n, self.r_n)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'")
