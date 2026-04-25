"""CuPy RawKernel ports of the four hot kernels in `_jit.py`.

Activated when `CFD_BACKEND=cupy`. Signatures mirror the Numba kernels so
the export block in `_jit.py` can swap implementations without changes
elsewhere. Bodies are direct transliterations of the Numba code, so the
parity tests in `tests/test_gpu_parity.py` should match the NumPy reference
(`_np_*` in `_jit.py`) at fp64 to ~1e-10.

This module imports `cupy` at the top — only import it from a context
where `USE_GPU` is True, otherwise it'll raise ImportError on machines
without CuPy installed.
"""
from __future__ import annotations

import cupy as cp  # type: ignore[import-not-found]
import numpy as np


_DIRICHLET = 0
_NEUMANN = 1


# ---------------------------------------------------------------------------
# apply_A: Poisson operator action p -> A p, BC-aware reads
# ---------------------------------------------------------------------------
_SRC_APPLY_A = r"""
extern "C" __global__ void apply_A(
    const double* __restrict__ p,
    double dx, double dy,
    int w, int e, int s, int n_,
    int ny, int nx,
    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double gw = (w == 0) ? -1.0 : 1.0;
    double ge = (e == 0) ? -1.0 : 1.0;
    double gs = (s == 0) ? -1.0 : 1.0;
    double gn = (n_ == 0) ? -1.0 : 1.0;

    double pc = p[j * nx + i];
    double pw_ = (i > 0)      ? p[j * nx + (i - 1)]   : gw * p[j * nx + 0];
    double pe_ = (i < nx - 1) ? p[j * nx + (i + 1)]   : ge * p[j * nx + (nx - 1)];
    double ps_ = (j > 0)      ? p[(j - 1) * nx + i]   : gs * p[0 * nx + i];
    double pn_ = (j < ny - 1) ? p[(j + 1) * nx + i]   : gn * p[(ny - 1) * nx + i];

    out[j * nx + i] = (pe_ - 2.0 * pc + pw_) * inv_dx2
                    + (pn_ - 2.0 * pc + ps_) * inv_dy2;
}
"""
_apply_A = cp.RawKernel(_SRC_APPLY_A, "apply_A")


def gpu_apply_A(p, dx, dy, w, e, s, n):
    ny, nx = p.shape
    out = cp.empty_like(p)
    block = (32, 8, 1)
    grid = ((nx + 31) // 32, (ny + 7) // 8, 1)
    _apply_A(
        grid, block,
        (p, np.float64(dx), np.float64(dy),
         np.int32(w), np.int32(e), np.int32(s), np.int32(n),
         np.int32(ny), np.int32(nx), out),
    )
    return out


# ---------------------------------------------------------------------------
# rb_gs: red-black Gauss-Seidel smoother. One CUDA kernel per color pass;
# the Python wrapper loops `n_sweeps` x {red, black} dispatches to mirror
# the existing Numba color order.
# ---------------------------------------------------------------------------
_SRC_RB_GS = r"""
extern "C" __global__ void rb_gs_pass(
    double* __restrict__ p,
    const double* __restrict__ rhs,
    const double* __restrict__ diag,
    double dx, double dy,
    int w, int e, int s, int n_,
    int parity,    // 0 = red ((i+j) even), 1 = black ((i+j) odd)
    int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    if (((i + j) & 1) != parity) return;

    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double gw = (w == 0) ? -1.0 : 1.0;
    double ge = (e == 0) ? -1.0 : 1.0;
    double gs = (s == 0) ? -1.0 : 1.0;
    double gn = (n_ == 0) ? -1.0 : 1.0;

    double pc = p[j * nx + i];
    double pw_ = (i > 0)      ? p[j * nx + (i - 1)] : gw * p[j * nx + 0];
    double pe_ = (i < nx - 1) ? p[j * nx + (i + 1)] : ge * p[j * nx + (nx - 1)];
    double ps_ = (j > 0)      ? p[(j - 1) * nx + i] : gs * p[0 * nx + i];
    double pn_ = (j < ny - 1) ? p[(j + 1) * nx + i] : gn * p[(ny - 1) * nx + i];

    double Ap = (pe_ - 2.0 * pc + pw_) * inv_dx2
              + (pn_ - 2.0 * pc + ps_) * inv_dy2;
    p[j * nx + i] = pc + (rhs[j * nx + i] - Ap) / diag[j * nx + i];
}
"""
_rb_gs_pass = cp.RawKernel(_SRC_RB_GS, "rb_gs_pass")


def gpu_rb_gs(p_in, rhs, dx, dy, n_sweeps, diag, w, e, s, n):
    ny, nx = p_in.shape
    p = p_in.copy()
    block = (32, 8, 1)
    grid = ((nx + 31) // 32, (ny + 7) // 8, 1)
    args_common = (np.float64(dx), np.float64(dy),
                   np.int32(w), np.int32(e), np.int32(s), np.int32(n))
    for _ in range(int(n_sweeps)):
        _rb_gs_pass(grid, block, (p, rhs, diag, *args_common,
                                  np.int32(0), np.int32(ny), np.int32(nx)))
        _rb_gs_pass(grid, block, (p, rhs, diag, *args_common,
                                  np.int32(1), np.int32(ny), np.int32(nx)))
    return p


# ---------------------------------------------------------------------------
# predictor: fused upwind advection + diffusion + pressure gradient + a_P
# ---------------------------------------------------------------------------
_SRC_PREDICTOR = r"""
extern "C" __global__ void predictor(
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ u_pad, const double* __restrict__ v_pad,
    const double* __restrict__ p_pad,
    const bool*   __restrict__ chi,
    double dt, double dx, double dy, double nu, double eta,
    double force_x, double force_y,
    int ny, int nx,
    double* __restrict__ u_star,
    double* __restrict__ v_star,
    double* __restrict__ a_P)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int Pnx = nx + 2;

    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double inv_dx2 = inv_dx * inv_dx;
    double inv_dy2 = inv_dy * inv_dy;
    double inv_2dx = 0.5 * inv_dx;
    double inv_2dy = 0.5 * inv_dy;
    double D_coef = 2.0 * inv_dx2 + 2.0 * inv_dy2;
    double nuD_dt = dt * nu * D_coef;
    double inv_eta = 1.0 / eta;

    double uc = u[j * nx + i];
    double vc = v[j * nx + i];

    // Pad indices: (j+1, i+1) is the cell center
    double u_e = u_pad[(j + 1) * Pnx + (i + 2)];
    double u_w = u_pad[(j + 1) * Pnx + i];
    double u_n = u_pad[(j + 2) * Pnx + (i + 1)];
    double u_s = u_pad[j       * Pnx + (i + 1)];
    double v_e = v_pad[(j + 1) * Pnx + (i + 2)];
    double v_w = v_pad[(j + 1) * Pnx + i];
    double v_n = v_pad[(j + 2) * Pnx + (i + 1)];
    double v_s = v_pad[j       * Pnx + (i + 1)];

    double dudx, dudy, dvdx, dvdy;
    if (uc > 0.0) {
        dudx = (uc - u_w) * inv_dx;
        dvdx = (vc - v_w) * inv_dx;
    } else {
        dudx = (u_e - uc) * inv_dx;
        dvdx = (v_e - vc) * inv_dx;
    }
    if (vc > 0.0) {
        dudy = (uc - u_s) * inv_dy;
        dvdy = (vc - v_s) * inv_dy;
    } else {
        dudy = (u_n - uc) * inv_dy;
        dvdy = (v_n - vc) * inv_dy;
    }
    double adv_u = uc * dudx + vc * dudy;
    double adv_v = uc * dvdx + vc * dvdy;

    double lap_u = (u_e - 2.0 * uc + u_w) * inv_dx2
                 + (u_n - 2.0 * uc + u_s) * inv_dy2;
    double lap_v = (v_e - 2.0 * vc + v_w) * inv_dx2
                 + (v_n - 2.0 * vc + v_s) * inv_dy2;
    double dif_u = nu * lap_u;
    double dif_v = nu * lap_v;

    double p_e = p_pad[(j + 1) * Pnx + (i + 2)];
    double p_w = p_pad[(j + 1) * Pnx + i];
    double p_n = p_pad[(j + 2) * Pnx + (i + 1)];
    double p_s = p_pad[j       * Pnx + (i + 1)];
    double gpx = (p_e - p_w) * inv_2dx;
    double gpy = (p_n - p_s) * inv_2dy;

    double chi_f = chi[j * nx + i] ? 1.0 : 0.0;
    double aP_ji = 1.0 + dt * chi_f * inv_eta + nuD_dt;
    double rhs_u_ji = uc + dt * (-adv_u + dif_u - gpx + force_x) + nuD_dt * uc;
    double rhs_v_ji = vc + dt * (-adv_v + dif_v - gpy + force_y) + nuD_dt * vc;
    u_star[j * nx + i] = rhs_u_ji / aP_ji;
    v_star[j * nx + i] = rhs_v_ji / aP_ji;
    a_P[j * nx + i]    = aP_ji;
}
"""
_predictor = cp.RawKernel(_SRC_PREDICTOR, "predictor")


def gpu_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta,
                  force_x, force_y):
    ny, nx = u.shape
    u_star = cp.empty_like(u)
    v_star = cp.empty_like(v)
    a_P = cp.empty_like(u)
    block = (32, 8, 1)
    grid = ((nx + 31) // 32, (ny + 7) // 8, 1)
    _predictor(
        grid, block,
        (u, v, u_pad, v_pad, p_pad, chi,
         np.float64(dt), np.float64(dx), np.float64(dy),
         np.float64(nu), np.float64(eta),
         np.float64(force_x), np.float64(force_y),
         np.int32(ny), np.int32(nx),
         u_star, v_star, a_P),
    )
    return u_star, v_star, a_P


# ---------------------------------------------------------------------------
# rc_faces_interior: two kernels (u-faces, v-faces). Caller pre-allocates
# u_f shape (ny, nx+1) and v_f shape (ny+1, nx); we write only the interior
# slices, mirroring the Numba kernel.
# ---------------------------------------------------------------------------
_SRC_RC_U = r"""
extern "C" __global__ void rc_faces_u(
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ p_pad, const double* __restrict__ a_P,
    double dt, double dx, double dy,
    int ny, int nx,
    double* __restrict__ u_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 1 .. nx-1
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. ny-1
    if (i < 1 || i >= nx || j >= ny) return;
    int Pnx = nx + 2;
    int Fnx = nx + 1;

    double inv_dx = 1.0 / dx;
    double inv_2dx = 0.5 * inv_dx;

    double uL = u[j * nx + (i - 1)];
    double uR = u[j * nx + i];
    double u_avg = 0.5 * (uL + uR);
    double a_face = 0.5 * (a_P[j * nx + (i - 1)] + a_P[j * nx + i]);

    double gpx_L = (p_pad[(j + 1) * Pnx + (i + 1)] - p_pad[(j + 1) * Pnx + (i - 1)]) * inv_2dx;
    double gpx_R = (p_pad[(j + 1) * Pnx + (i + 2)] - p_pad[(j + 1) * Pnx + i])       * inv_2dx;
    double gpx_face_avg = 0.5 * (gpx_L + gpx_R);
    double gpx_face_compact =
        (p_pad[(j + 1) * Pnx + (i + 1)] - p_pad[(j + 1) * Pnx + i]) * inv_dx;

    u_f[j * Fnx + i] = u_avg - (dt / a_face) * (gpx_face_compact - gpx_face_avg);
}
"""
_rc_faces_u = cp.RawKernel(_SRC_RC_U, "rc_faces_u")

_SRC_RC_V = r"""
extern "C" __global__ void rc_faces_v(
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ p_pad, const double* __restrict__ a_P,
    double dt, double dx, double dy,
    int ny, int nx,
    double* __restrict__ v_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. nx-1
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // 1 .. ny-1
    if (i >= nx || j < 1 || j >= ny) return;
    int Pnx = nx + 2;

    double inv_dy = 1.0 / dy;
    double inv_2dy = 0.5 * inv_dy;

    double vD = v[(j - 1) * nx + i];
    double vU = v[j * nx + i];
    double v_avg = 0.5 * (vD + vU);
    double a_face_y = 0.5 * (a_P[(j - 1) * nx + i] + a_P[j * nx + i]);

    double gpy_D = (p_pad[(j + 1) * Pnx + (i + 1)] - p_pad[(j - 1) * Pnx + (i + 1)]) * inv_2dy;
    double gpy_U = (p_pad[(j + 2) * Pnx + (i + 1)] - p_pad[j       * Pnx + (i + 1)]) * inv_2dy;
    double gpy_face_avg = 0.5 * (gpy_D + gpy_U);
    double gpy_face_compact =
        (p_pad[(j + 1) * Pnx + (i + 1)] - p_pad[j * Pnx + (i + 1)]) * inv_dy;

    v_f[j * nx + i] = v_avg - (dt / a_face_y) * (gpy_face_compact - gpy_face_avg);
}
"""
_rc_faces_v = cp.RawKernel(_SRC_RC_V, "rc_faces_v")


def gpu_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f, v_f):
    ny, nx = u.shape
    block = (32, 8, 1)
    grid = ((nx + 31) // 32, (ny + 7) // 8, 1)
    args_scalars = (np.float64(dt), np.float64(dx), np.float64(dy),
                    np.int32(ny), np.int32(nx))
    _rc_faces_u(grid, block, (u, v, p_pad, a_P, *args_scalars, u_f))
    _rc_faces_v(grid, block, (u, v, p_pad, a_P, *args_scalars, v_f))
