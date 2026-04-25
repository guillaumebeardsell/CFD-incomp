"""Central-difference viscous fluxes for 2D compressible Navier-Stokes.

All fluxes are computed at cell faces using simple centred gradients. The
staircase immersed boundary (IB) model is used: primitive velocity is zeroed
inside solid cells before gradient evaluation so that fluid/solid faces see
a proper no-slip velocity jump. Heat flux at walls uses fluid-side temperature.

Sign convention matches the inviscid flux: the returned arrays are subtracted
from the inviscid flux in the divergence formula (see solver._compute_rhs).
"""
import numpy as np

from . import config


def sutherland_mu(T):
    return (
        config.MU_REF
        * (T / config.T_SUTH_REF) ** 1.5
        * (config.T_SUTH_REF + config.S_SUTH)
        / (T + config.S_SUTH)
    )


def _viscous_coeffs(T_face, gamma, R):
    mu = sutherland_mu(T_face)
    k = mu * gamma * R / ((gamma - 1.0) * config.PR)
    return mu, k


def viscous_flux_x(W, dx, dy, gamma, R):
    """Viscous flux at x-normal faces, returned shape (4, ny, nx+1).

    W has shape (4, ny+4, nx+4) with ghost cells populated by BCs.
    """
    rho, u, v, p = W[0], W[1], W[2], W[3]
    T = p / (rho * R)

    nyp, nxp = W.shape[1], W.shape[2]  # nyp = ny+4, nxp = nx+4
    yI = slice(2, nyp - 2)             # interior y rows, length ny
    xL_slice = slice(1, nxp - 2)       # left-cell columns for each x-face, length nx+1
    xR_slice = slice(2, nxp - 1)       # right-cell columns,                length nx+1
    xLm1 = slice(0, nxp - 3)           # one column to the left of xL_slice (unused below)
    xRp1 = slice(3, nxp)               # one column to the right of xR_slice (unused below)

    u_L = u[yI, xL_slice]; u_R = u[yI, xR_slice]
    v_L = v[yI, xL_slice]; v_R = v[yI, xR_slice]
    T_L = T[yI, xL_slice]; T_R = T[yI, xR_slice]

    du_dx = (u_R - u_L) / dx
    dv_dx = (v_R - v_L) / dx
    dT_dx = (T_R - T_L) / dx

    # Normal (y) gradient at each x-face: centred 2-point difference using rows above/below.
    yBot = slice(1, nyp - 3)
    yTop = slice(3, nyp - 1)
    u_bot = 0.5 * (u[yBot, xL_slice] + u[yBot, xR_slice])
    u_top = 0.5 * (u[yTop, xL_slice] + u[yTop, xR_slice])
    v_bot = 0.5 * (v[yBot, xL_slice] + v[yBot, xR_slice])
    v_top = 0.5 * (v[yTop, xL_slice] + v[yTop, xR_slice])
    du_dy = (u_top - u_bot) / (2.0 * dy)
    dv_dy = (v_top - v_bot) / (2.0 * dy)

    u_face = 0.5 * (u_L + u_R)
    v_face = 0.5 * (v_L + v_R)
    T_face = 0.5 * (T_L + T_R)
    mu, k = _viscous_coeffs(T_face, gamma, R)

    div = du_dx + dv_dy
    tau_xx = mu * (2.0 * du_dx - (2.0 / 3.0) * div)
    tau_xy = mu * (du_dy + dv_dx)
    q_x = -k * dT_dx

    G0 = np.zeros_like(u_face)
    G1 = tau_xx
    G2 = tau_xy
    G3 = u_face * tau_xx + v_face * tau_xy - q_x
    return np.stack([G0, G1, G2, G3], axis=0)


def viscous_flux_y(W, dx, dy, gamma, R):
    """Viscous flux at y-normal faces, returned shape (4, ny+1, nx)."""
    rho, u, v, p = W[0], W[1], W[2], W[3]
    T = p / (rho * R)

    nyp, nxp = W.shape[1], W.shape[2]
    xI = slice(2, nxp - 2)
    yB_slice = slice(1, nyp - 2)
    yT_slice = slice(2, nyp - 1)

    u_B = u[yB_slice, xI]; u_T = u[yT_slice, xI]
    v_B = v[yB_slice, xI]; v_T = v[yT_slice, xI]
    T_B = T[yB_slice, xI]; T_T = T[yT_slice, xI]

    du_dy = (u_T - u_B) / dy
    dv_dy = (v_T - v_B) / dy
    dT_dy = (T_T - T_B) / dy

    xLeft = slice(1, nxp - 3)
    xRight = slice(3, nxp - 1)
    u_left = 0.5 * (u[yB_slice, xLeft] + u[yT_slice, xLeft])
    u_right = 0.5 * (u[yB_slice, xRight] + u[yT_slice, xRight])
    v_left = 0.5 * (v[yB_slice, xLeft] + v[yT_slice, xLeft])
    v_right = 0.5 * (v[yB_slice, xRight] + v[yT_slice, xRight])
    du_dx = (u_right - u_left) / (2.0 * dx)
    dv_dx = (v_right - v_left) / (2.0 * dx)

    u_face = 0.5 * (u_B + u_T)
    v_face = 0.5 * (v_B + v_T)
    T_face = 0.5 * (T_B + T_T)
    mu, k = _viscous_coeffs(T_face, gamma, R)

    div = du_dx + dv_dy
    tau_yy = mu * (2.0 * dv_dy - (2.0 / 3.0) * div)
    tau_xy = mu * (du_dy + dv_dx)
    q_y = -k * dT_dy

    G0 = np.zeros_like(u_face)
    G1 = tau_xy
    G2 = tau_yy
    G3 = u_face * tau_xy + v_face * tau_yy - q_y
    return np.stack([G0, G1, G2, G3], axis=0)
