"""Implicit residual smoothing via ADI tridiagonal sweeps.

Solves (1 - eps * d2/dx2)(1 - eps * d2/dy2) R_smooth = R for each component,
using the Thomas algorithm along each axis with Neumann end conditions.
"""
from __future__ import annotations

import numpy as np


def _thomas_neumann(rhs, eps):
    """Solve the tridiagonal system (-eps, 1+2*eps, -eps) R = rhs along axis=-1
    with zero-gradient Neumann ends (R_{-1}=R_0, R_N=R_{N-1}), which folds the
    off-boundary coefficient into the diagonal: diag[0] = diag[-1] = 1+eps.

    rhs: (..., n) array. Returns (..., n).
    """
    n = rhs.shape[-1]
    # Coefficient arrays per row (same for every row, but we broadcast).
    a = -eps  # sub-diagonal
    c = -eps  # super-diagonal
    b = np.full(n, 1.0 + 2.0 * eps)
    b[0] = 1.0 + eps
    b[-1] = 1.0 + eps

    # Forward sweep (Thomas). c' and d' are modified in place.
    cp = np.empty(n)
    dp = np.empty_like(rhs)
    cp[0] = c / b[0]
    dp[..., 0] = rhs[..., 0] / b[0]
    for i in range(1, n):
        denom = b[i] - a * cp[i - 1]
        cp[i] = c / denom if i < n - 1 else 0.0
        dp[..., i] = (rhs[..., i] - a * dp[..., i - 1]) / denom

    # Back substitution.
    out = np.empty_like(rhs)
    out[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        out[..., i] = dp[..., i] - cp[i] * out[..., i + 1]
    return out


def smooth_residual(rhs, eps):
    """Apply IRS to a (4, ny, nx) residual array. Returns the smoothed array.

    Factored ADI: sweep along x, then along y. Each sweep uses Neumann ends.
    """
    if eps <= 0.0:
        return rhs
    # X-sweep: last axis is nx -> operate directly.
    r = _thomas_neumann(rhs, eps)
    # Y-sweep: move ny to last axis, solve, swap back.
    r = np.swapaxes(r, 1, 2)          # (4, nx, ny)
    r = _thomas_neumann(r, eps)
    r = np.swapaxes(r, 1, 2)          # (4, ny, nx)
    return r
