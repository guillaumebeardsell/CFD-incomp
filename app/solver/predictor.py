import numpy as np

from ._jit import predictor_kernel
from ._xp import xp


def predictor(u, v, p, chi, dt, dx, dy, nu, eta, bcs,
              force_x: float = 0.0, force_y: float = 0.0):
    """Semi-implicit momentum predictor with Brinkman penalization.

    Point-implicit diffusion: the diagonal of the viscous Laplacian is
    absorbed into a_P so the explicit viscous dt-cap relaxes. Off-diagonal
    neighbors are still evaluated at the old time level (first-order).

    Returns (u_star, v_star, a_P).
    a_P = 1 + dt*chi/eta + dt*nu*(2/dx^2 + 2/dy^2).
    """
    u_pad, v_pad = bcs.pad_velocity(u, v)
    p_pad = bcs.pad_pressure(p)
    chi_b = xp.asarray(chi, dtype=xp.bool_)
    return predictor_kernel(
        u, v, u_pad, v_pad, p_pad, chi_b, dt, dx, dy, nu, eta,
        float(force_x), float(force_y),
    )
