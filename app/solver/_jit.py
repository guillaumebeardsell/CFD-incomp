"""Numba-compiled hot-path kernels with a NumPy fallback.

Exports four module-level names that either point at an `@njit` kernel or
at a pure-NumPy reference implementation:

    apply_A_kernel    - Poisson operator action p -> A·p, bc-aware reads
    rb_gs_kernel      - red-black Gauss-Seidel smoother (in-place updates)
    predictor_kernel  - fused upwind-advection + diffusion + rhs + a_P
    rc_faces_kernel   - fused Rhie-Chow interior face velocities

Fallback is triggered by either `numba` being unavailable or
`CFD_DISABLE_JIT=1` in the environment. The fallback path preserves
bit-for-bit behavior of the prior NumPy implementation.

DIRICHLET = 0, NEUMANN = 1 match poisson_mg module constants.
"""
from __future__ import annotations

import os
import numpy as np

_DIRICHLET = 0  # ghost = -interior (phi=0 at face)
_NEUMANN = 1    # ghost = +interior (d phi/dn = 0)


_DISABLE = os.environ.get("CFD_DISABLE_JIT") == "1"
try:
    if _DISABLE:
        raise ImportError("CFD_DISABLE_JIT=1")
    from numba import njit
    USE_JIT = True
except ImportError:
    USE_JIT = False

    def njit(*args, **kwargs):
        """No-op decorator when Numba is unavailable."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator


# ---------------------------------------------------------------------------
# NumPy reference kernels (used when USE_JIT is False). These are copies of
# the original logic from poisson_mg.py, predictor.py, operators.py.
# ---------------------------------------------------------------------------

def _np_pad_poisson(p, w, e, s, n):
    ny, nx = p.shape
    out = np.empty((ny + 2, nx + 2), dtype=p.dtype)
    out[1:-1, 1:-1] = p
    out[1:-1, 0]  = -p[:, 0]  if w == _DIRICHLET else p[:, 0]
    out[1:-1, -1] = -p[:, -1] if e == _DIRICHLET else p[:, -1]
    out[0, 1:-1]  = -p[0, :]  if s == _DIRICHLET else p[0, :]
    out[-1, 1:-1] = -p[-1, :] if n == _DIRICHLET else p[-1, :]
    out[0, 0] = out[0, 1]; out[0, -1] = out[0, -2]
    out[-1, 0] = out[-1, 1]; out[-1, -1] = out[-1, -2]
    return out


def _np_apply_A(p, dx, dy, w, e, s, n):
    pp = _np_pad_poisson(p, w, e, s, n)
    return (
        (pp[1:-1, 2:] - 2.0 * pp[1:-1, 1:-1] + pp[1:-1, :-2]) / dx ** 2
        + (pp[2:, 1:-1] - 2.0 * pp[1:-1, 1:-1] + pp[:-2, 1:-1]) / dy ** 2
    )


def _np_rb_gs(p_in, rhs, dx, dy, n_sweeps, diag, w, e, s, n):
    p = p_in.copy()
    for _ in range(n_sweeps):
        r = rhs - _np_apply_A(p, dx, dy, w, e, s, n)
        p[0::2, 0::2] += r[0::2, 0::2] / diag[0::2, 0::2]
        p[1::2, 1::2] += r[1::2, 1::2] / diag[1::2, 1::2]
        r = rhs - _np_apply_A(p, dx, dy, w, e, s, n)
        p[0::2, 1::2] += r[0::2, 1::2] / diag[0::2, 1::2]
        p[1::2, 0::2] += r[1::2, 0::2] / diag[1::2, 0::2]
    return p


def _np_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta,
                  force_x, force_y):
    # advection (upwind)
    u_fwd_x = (u_pad[1:-1, 2:]  - u) / dx
    u_bwd_x = (u - u_pad[1:-1, :-2]) / dx
    u_fwd_y = (u_pad[2:, 1:-1]  - u) / dy
    u_bwd_y = (u - u_pad[:-2, 1:-1]) / dy
    v_fwd_x = (v_pad[1:-1, 2:]  - v) / dx
    v_bwd_x = (v - v_pad[1:-1, :-2]) / dx
    v_fwd_y = (v_pad[2:, 1:-1]  - v) / dy
    v_bwd_y = (v - v_pad[:-2, 1:-1]) / dy
    dudx = np.where(u > 0, u_bwd_x, u_fwd_x)
    dudy = np.where(v > 0, u_bwd_y, u_fwd_y)
    dvdx = np.where(u > 0, v_bwd_x, v_fwd_x)
    dvdy = np.where(v > 0, v_bwd_y, v_fwd_y)
    adv_u = u * dudx + v * dudy
    adv_v = u * dvdx + v * dvdy
    # diffusion
    lap_u = ((u_pad[1:-1, 2:] - 2.0 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dx ** 2
            + (u_pad[2:, 1:-1] - 2.0 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dy ** 2)
    lap_v = ((v_pad[1:-1, 2:] - 2.0 * v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dx ** 2
            + (v_pad[2:, 1:-1] - 2.0 * v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dy ** 2)
    dif_u = nu * lap_u
    dif_v = nu * lap_v
    # pressure gradient
    gpx = (p_pad[1:-1, 2:] - p_pad[1:-1, :-2]) / (2.0 * dx)
    gpy = (p_pad[2:, 1:-1] - p_pad[:-2, 1:-1]) / (2.0 * dy)
    # implicit diffusion diagonal and a_P
    D_coef = 2.0 / dx ** 2 + 2.0 / dy ** 2
    nuD_dt = dt * nu * D_coef
    a_P = 1.0 + dt * chi.astype(np.float64) / eta + nuD_dt
    rhs_u = u + dt * (-adv_u + dif_u - gpx + force_x) + nuD_dt * u
    rhs_v = v + dt * (-adv_v + dif_v - gpy + force_y) + nuD_dt * v
    u_star = rhs_u / a_P
    v_star = rhs_v / a_P
    return u_star, v_star, a_P


def _np_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f, v_f):
    """Fill u_f[:, 1:-1] and v_f[1:-1, :]. Boundary faces handled by caller."""
    gpx = (p_pad[1:-1, 2:] - p_pad[1:-1, :-2]) / (2.0 * dx)
    gpy = (p_pad[2:, 1:-1] - p_pad[:-2, 1:-1]) / (2.0 * dy)
    u_avg = 0.5 * (u[:, :-1] + u[:, 1:])
    a_face = 0.5 * (a_P[:, :-1] + a_P[:, 1:])
    gpx_face_compact = (p_pad[1:-1, 2:-1] - p_pad[1:-1, 1:-2]) / dx
    gpx_face_avg = 0.5 * (gpx[:, :-1] + gpx[:, 1:])
    u_f[:, 1:-1] = u_avg - (dt / a_face) * (gpx_face_compact - gpx_face_avg)
    v_avg = 0.5 * (v[:-1, :] + v[1:, :])
    a_face_y = 0.5 * (a_P[:-1, :] + a_P[1:, :])
    gpy_face_compact = (p_pad[2:-1, 1:-1] - p_pad[1:-2, 1:-1]) / dy
    gpy_face_avg = 0.5 * (gpy[:-1, :] + gpy[1:, :])
    v_f[1:-1, :] = v_avg - (dt / a_face_y) * (gpy_face_compact - gpy_face_avg)


# ---------------------------------------------------------------------------
# Numba kernels. Signatures accept 4 separate ints (w, e, s, n) instead of a
# tuple — portable across Numba versions.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, boundscheck=False)
def _jit_apply_A(p, dx, dy, w, e, s, n):
    ny, nx = p.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    # Ghost sign: Dirichlet -> -1 (ghost = -interior), Neumann -> +1
    gw = -1.0 if w == _DIRICHLET else 1.0
    ge = -1.0 if e == _DIRICHLET else 1.0
    gs = -1.0 if s == _DIRICHLET else 1.0
    gn = -1.0 if n == _DIRICHLET else 1.0
    out = np.empty_like(p)
    for j in range(ny):
        for i in range(nx):
            pc = p[j, i]
            pw_ = p[j, i - 1] if i > 0      else gw * p[j, 0]
            pe_ = p[j, i + 1] if i < nx - 1 else ge * p[j, nx - 1]
            ps_ = p[j - 1, i] if j > 0      else gs * p[0, i]
            pn_ = p[j + 1, i] if j < ny - 1 else gn * p[ny - 1, i]
            out[j, i] = (pe_ - 2.0 * pc + pw_) * inv_dx2 + (pn_ - 2.0 * pc + ps_) * inv_dy2
    return out


@njit(cache=True, fastmath=True, boundscheck=False)
def _jit_rb_gs(p_in, rhs, dx, dy, n_sweeps, diag, w, e, s, n):
    ny, nx = p_in.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    gw = -1.0 if w == _DIRICHLET else 1.0
    ge = -1.0 if e == _DIRICHLET else 1.0
    gs = -1.0 if s == _DIRICHLET else 1.0
    gn = -1.0 if n == _DIRICHLET else 1.0
    p = p_in.copy()
    for _ in range(n_sweeps):
        # Red pass: (i+j) even
        for j in range(ny):
            i_start = 0 if (j & 1) == 0 else 1
            for i in range(i_start, nx, 2):
                pc = p[j, i]
                pw_ = p[j, i - 1] if i > 0      else gw * p[j, 0]
                pe_ = p[j, i + 1] if i < nx - 1 else ge * p[j, nx - 1]
                ps_ = p[j - 1, i] if j > 0      else gs * p[0, i]
                pn_ = p[j + 1, i] if j < ny - 1 else gn * p[ny - 1, i]
                Ap = (pe_ - 2.0 * pc + pw_) * inv_dx2 + (pn_ - 2.0 * pc + ps_) * inv_dy2
                p[j, i] = pc + (rhs[j, i] - Ap) / diag[j, i]
        # Black pass: (i+j) odd
        for j in range(ny):
            i_start = 1 if (j & 1) == 0 else 0
            for i in range(i_start, nx, 2):
                pc = p[j, i]
                pw_ = p[j, i - 1] if i > 0      else gw * p[j, 0]
                pe_ = p[j, i + 1] if i < nx - 1 else ge * p[j, nx - 1]
                ps_ = p[j - 1, i] if j > 0      else gs * p[0, i]
                pn_ = p[j + 1, i] if j < ny - 1 else gn * p[ny - 1, i]
                Ap = (pe_ - 2.0 * pc + pw_) * inv_dx2 + (pn_ - 2.0 * pc + ps_) * inv_dy2
                p[j, i] = pc + (rhs[j, i] - Ap) / diag[j, i]
    return p


@njit(cache=True, fastmath=True, boundscheck=False)
def _jit_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta,
                   force_x, force_y):
    ny, nx = u.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    inv_2dx = 0.5 * inv_dx
    inv_2dy = 0.5 * inv_dy
    D_coef = 2.0 * inv_dx2 + 2.0 * inv_dy2
    nuD_dt = dt * nu * D_coef
    inv_eta = 1.0 / eta
    u_star = np.empty_like(u)
    v_star = np.empty_like(v)
    a_P = np.empty_like(u)
    for j in range(ny):
        for i in range(nx):
            uc = u[j, i]
            vc = v[j, i]
            # Pad indices: p_pad[j+1, i+1] == cell center
            u_e = u_pad[j + 1, i + 2]; u_w = u_pad[j + 1, i]
            u_n = u_pad[j + 2, i + 1]; u_s = u_pad[j,     i + 1]
            v_e = v_pad[j + 1, i + 2]; v_w = v_pad[j + 1, i]
            v_n = v_pad[j + 2, i + 1]; v_s = v_pad[j,     i + 1]
            # Upwind advection
            if uc > 0.0:
                dudx = (uc - u_w) * inv_dx
                dvdx = (vc - v_w) * inv_dx
            else:
                dudx = (u_e - uc) * inv_dx
                dvdx = (v_e - vc) * inv_dx
            if vc > 0.0:
                dudy = (uc - u_s) * inv_dy
                dvdy = (vc - v_s) * inv_dy
            else:
                dudy = (u_n - uc) * inv_dy
                dvdy = (v_n - vc) * inv_dy
            adv_u = uc * dudx + vc * dudy
            adv_v = uc * dvdx + vc * dvdy
            # Diffusion (explicit neighbors; diagonal absorbed into a_P)
            lap_u = (u_e - 2.0 * uc + u_w) * inv_dx2 + (u_n - 2.0 * uc + u_s) * inv_dy2
            lap_v = (v_e - 2.0 * vc + v_w) * inv_dx2 + (v_n - 2.0 * vc + v_s) * inv_dy2
            dif_u = nu * lap_u
            dif_v = nu * lap_v
            # Pressure gradient
            p_e = p_pad[j + 1, i + 2]; p_w = p_pad[j + 1, i]
            p_n = p_pad[j + 2, i + 1]; p_s = p_pad[j,     i + 1]
            gpx = (p_e - p_w) * inv_2dx
            gpy = (p_n - p_s) * inv_2dy
            # Brinkman + implicit-diffusion diagonal in a_P
            chi_f = 1.0 if chi[j, i] else 0.0
            aP_ji = 1.0 + dt * chi_f * inv_eta + nuD_dt
            rhs_u_ji = uc + dt * (-adv_u + dif_u - gpx + force_x) + nuD_dt * uc
            rhs_v_ji = vc + dt * (-adv_v + dif_v - gpy + force_y) + nuD_dt * vc
            u_star[j, i] = rhs_u_ji / aP_ji
            v_star[j, i] = rhs_v_ji / aP_ji
            a_P[j, i] = aP_ji
    return u_star, v_star, a_P


@njit(cache=True, fastmath=True, boundscheck=False)
def _jit_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f, v_f):
    """Fill u_f[:, 1:nx] and v_f[1:ny, :]. Boundary faces handled by caller."""
    ny, nx = u.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_2dx = 0.5 * inv_dx
    inv_2dy = 0.5 * inv_dy
    # u_f interior faces: i in 1..nx-1 (nx-1 faces between cells i-1 and i)
    for j in range(ny):
        for i in range(1, nx):
            uL = u[j, i - 1]
            uR = u[j, i]
            u_avg = 0.5 * (uL + uR)
            a_face = 0.5 * (a_P[j, i - 1] + a_P[j, i])
            # Central gradients at cells i-1 and i
            gpx_L = (p_pad[j + 1, i + 1] - p_pad[j + 1, i - 1]) * inv_2dx
            gpx_R = (p_pad[j + 1, i + 2] - p_pad[j + 1, i])     * inv_2dx
            gpx_face_avg = 0.5 * (gpx_L + gpx_R)
            gpx_face_compact = (p_pad[j + 1, i + 1] - p_pad[j + 1, i]) * inv_dx
            u_f[j, i] = u_avg - (dt / a_face) * (gpx_face_compact - gpx_face_avg)
    # v_f interior faces: j in 1..ny-1
    for j in range(1, ny):
        for i in range(nx):
            vD = v[j - 1, i]
            vU = v[j, i]
            v_avg = 0.5 * (vD + vU)
            a_face_y = 0.5 * (a_P[j - 1, i] + a_P[j, i])
            gpy_D = (p_pad[j + 1, i + 1] - p_pad[j - 1, i + 1]) * inv_2dy
            gpy_U = (p_pad[j + 2, i + 1] - p_pad[j,     i + 1]) * inv_2dy
            gpy_face_avg = 0.5 * (gpy_D + gpy_U)
            gpy_face_compact = (p_pad[j + 1, i + 1] - p_pad[j, i + 1]) * inv_dy
            v_f[j, i] = v_avg - (dt / a_face_y) * (gpy_face_compact - gpy_face_avg)


# ---------------------------------------------------------------------------
# Exported kernel symbols. Callers in poisson_mg/predictor/operators use
# these names and never touch USE_JIT / USE_GPU directly.
# ---------------------------------------------------------------------------

from ._xp import USE_GPU

if USE_GPU:
    from ._cupy_kernels import (
        gpu_apply_A, gpu_rb_gs, gpu_predictor, gpu_rc_faces_interior,
    )
    apply_A_kernel = gpu_apply_A
    rb_gs_kernel = gpu_rb_gs
    predictor_kernel = gpu_predictor
    rc_faces_interior_kernel = gpu_rc_faces_interior
elif USE_JIT:
    apply_A_kernel = _jit_apply_A
    rb_gs_kernel = _jit_rb_gs
    predictor_kernel = _jit_predictor
    rc_faces_interior_kernel = _jit_rc_faces_interior
else:
    apply_A_kernel = _np_apply_A
    rb_gs_kernel = _np_rb_gs
    predictor_kernel = _np_predictor
    rc_faces_interior_kernel = _np_rc_faces_interior
