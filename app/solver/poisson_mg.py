"""Geometric multigrid V-cycle for Poisson equations on a rectangular domain.

Parameterized by BC type per side (Dirichlet-zero or Neumann). Used for:
  - pressure-correction  (east Dirichlet, others Neumann)
  - streamfunction       (all four Dirichlet)
"""
import numpy as np

from ._jit import apply_A_kernel, rb_gs_kernel
from ._xp import xp, asarray, asnumpy

DIRICHLET = 0   # ghost = -interior  (phi=0 at face)
NEUMANN = 1     # ghost =  interior  (d phi/dn = 0)


def apply_A(p, dx, dy, bc_sides):
    w, e, s, n = bc_sides
    return apply_A_kernel(p, dx, dy, w, e, s, n)


def _diag(shape, dx, dy, bc_sides):
    ny, nx = shape
    d = xp.full((ny, nx), -2.0 / dx ** 2 - 2.0 / dy ** 2)
    w, e, s, n = bc_sides
    d[:, 0]  += (+1.0 if w == NEUMANN else -1.0) / dx ** 2
    d[:, -1] += (+1.0 if e == NEUMANN else -1.0) / dx ** 2
    d[0, :]  += (+1.0 if s == NEUMANN else -1.0) / dy ** 2
    d[-1, :] += (+1.0 if n == NEUMANN else -1.0) / dy ** 2
    return d


def _jacobi(p, rhs, dx, dy, n_sweeps, omega, diag, bc_sides):
    for _ in range(n_sweeps):
        p = p + omega * (rhs - apply_A(p, dx, dy, bc_sides)) / diag
    return p


def _rb_gs(p, rhs, dx, dy, n_sweeps, diag, bc_sides):
    """Red-black Gauss-Seidel smoother; delegates to JIT / NumPy kernel."""
    w, e, s, n = bc_sides
    return rb_gs_kernel(p, rhs, dx, dy, n_sweeps, diag, w, e, s, n)


def _restrict(fine):
    return 0.25 * (fine[::2, ::2] + fine[1::2, ::2] + fine[::2, 1::2] + fine[1::2, 1::2])


def _prolong(coarse):
    return xp.repeat(xp.repeat(coarse, 2, axis=0), 2, axis=1)


def _can_coarsen(shape):
    return shape[0] % 2 == 0 and shape[1] % 2 == 0 and shape[0] >= 8 and shape[1] >= 8


_COARSE_LU_CACHE: dict = {}


def _build_coarse_matrix(ny, nx, dx, dy, bc_sides):
    n = ny * nx
    A = np.zeros((n, n))
    w, e, s, n_ = bc_sides
    inv_dx2 = 1.0 / dx ** 2
    inv_dy2 = 1.0 / dy ** 2
    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            diag = -2.0 * inv_dx2 - 2.0 * inv_dy2
            if i > 0:    A[row, row - 1] += inv_dx2
            else:        diag += (+1.0 if w == NEUMANN else -1.0) * inv_dx2
            if i < nx-1: A[row, row + 1] += inv_dx2
            else:        diag += (+1.0 if e == NEUMANN else -1.0) * inv_dx2
            if j > 0:    A[row, row - nx] += inv_dy2
            else:        diag += (+1.0 if s == NEUMANN else -1.0) * inv_dy2
            if j < ny-1: A[row, row + nx] += inv_dy2
            else:        diag += (+1.0 if n_ == NEUMANN else -1.0) * inv_dy2
            A[row, row] = diag
    pure_neumann = (w == NEUMANN and e == NEUMANN and s == NEUMANN and n_ == NEUMANN)
    if pure_neumann:
        A[0, :] = 0.0
        A[0, 0] = 1.0
    return A, pure_neumann


def _coarse_solve(rhs, dx, dy, bc_sides):
    """Direct LU solve at the coarsest level. The (matrix, LU) pair is cached
    per (shape, dx, dy, bc_sides) so we don't rebuild it every V-cycle."""
    from scipy.linalg import lu_factor, lu_solve
    ny, nx = rhs.shape
    key = (ny, nx, dx, dy, bc_sides)
    entry = _COARSE_LU_CACHE.get(key)
    if entry is None:
        A, pure_neumann = _build_coarse_matrix(ny, nx, dx, dy, bc_sides)
        lu_piv = lu_factor(A)
        entry = (lu_piv, pure_neumann)
        _COARSE_LU_CACHE[key] = entry
    lu_piv, pure_neumann = entry
    b = asnumpy(rhs).ravel().copy()
    if pure_neumann:
        b[0] = 0.0
    x = lu_solve(lu_piv, b)
    return asarray(x.reshape(ny, nx))


def _vcycle(p, rhs, dx, dy, bc_sides, n_smooth=2, omega=0.8):
    diag = _diag(p.shape, dx, dy, bc_sides)
    if not _can_coarsen(p.shape):
        return _coarse_solve(rhs, dx, dy, bc_sides)
    p = _rb_gs(p, rhs, dx, dy, n_smooth, diag, bc_sides)
    r = rhs - apply_A(p, dx, dy, bc_sides)
    rc = _restrict(r)
    ec = xp.zeros_like(rc)
    ec = _vcycle(ec, rc, dx * 2, dy * 2, bc_sides, n_smooth, omega)
    p = p + _prolong(ec)
    p = _rb_gs(p, rhs, dx, dy, n_smooth, diag, bc_sides)
    return p


def _fmg_init(rhs, dx, dy, bc_sides, n_smooth=2):
    """Full-multigrid (nested iteration) initial guess: coarse-solve, prolong,
    one V-cycle, and repeat up to the fine grid. Produces a near-converged
    starting field at roughly one fine-grid V-cycle of total cost, so cold-start
    Poisson solves need far fewer outer V-cycles to hit tolerance."""
    if not _can_coarsen(rhs.shape):
        return _coarse_solve(rhs, dx, dy, bc_sides)
    rc = _restrict(rhs)
    ec = _fmg_init(rc, dx * 2, dy * 2, bc_sides, n_smooth)
    p = _prolong(ec)
    return _vcycle(p, rhs, dx, dy, bc_sides, n_smooth=n_smooth)


# Standard BC presets
BC_PCORR = (NEUMANN, DIRICHLET, NEUMANN, NEUMANN)   # west, east, south, north
BC_PSI   = (DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET)


def solve(rhs, dx, dy, tol=1e-4, max_cycles=30, n_smooth=2, p0=None,
          bc_sides=BC_PCORR):
    p = _fmg_init(rhs, dx, dy, bc_sides, n_smooth) if p0 is None else p0.copy()
    rhs_norm = float(xp.linalg.norm(rhs)) + 1e-30
    rel = 1.0
    for c in range(max_cycles):
        p = _vcycle(p, rhs, dx, dy, bc_sides, n_smooth=n_smooth)
        rel = float(xp.linalg.norm(rhs - apply_A(p, dx, dy, bc_sides))) / rhs_norm
        if rel < tol:
            return p, c + 1, rel
    return p, max_cycles, rel
