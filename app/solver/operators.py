import numpy as np

from ._jit import rc_faces_interior_kernel
from ._xp import xp


def gradient(field_pad: np.ndarray, dx: float, dy: float):
    """Central-difference gradient on interior cells from a (ny+2, nx+2) padded array."""
    gx = (field_pad[1:-1, 2:] - field_pad[1:-1, :-2]) / (2.0 * dx)
    gy = (field_pad[2:, 1:-1] - field_pad[:-2, 1:-1]) / (2.0 * dy)
    return gx, gy


def laplacian(field_pad: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (field_pad[1:-1, 2:] - 2.0 * field_pad[1:-1, 1:-1] + field_pad[1:-1, :-2]) / dx ** 2
        + (field_pad[2:, 1:-1] - 2.0 * field_pad[1:-1, 1:-1] + field_pad[:-2, 1:-1]) / dy ** 2
    )


def divergence_from_faces(u_face_e: np.ndarray, v_face_n: np.ndarray,
                          dx: float, dy: float) -> np.ndarray:
    """Cell-centered divergence from face velocities.
    u_face_e: (ny, nx+1), v_face_n: (ny+1, nx)."""
    return (u_face_e[:, 1:] - u_face_e[:, :-1]) / dx + (v_face_n[1:, :] - v_face_n[:-1, :]) / dy


def rhie_chow_faces(u, v, p_pad, a_P, dt, dx, dy, bcs):
    """Rhie-Chow face velocities with Brinkman-aware a_P.

    Returns (u_face_e, v_face_n):
      u_face_e shape (ny, nx+1): east-face x-velocity
      v_face_n shape (ny+1, nx): north-face y-velocity
    Boundary faces come from BCs; interior faces use RC kernel (JIT/NumPy).
    """
    ny, nx = u.shape
    u_f = xp.empty((ny, nx + 1), dtype=u.dtype)
    v_f = xp.empty((ny + 1, nx), dtype=v.dtype)

    rc_faces_interior_kernel(u, v, p_pad, a_P, dt, dx, dy, u_f, v_f)

    # Boundary x-faces (U_in is scalar or (ny,) profile; NumPy broadcasts)
    u_f[:, 0] = bcs.U_in                 # inlet Dirichlet
    u_f[:, -1] = u[:, -1]                # outlet Neumann
    # Top/bottom v-face: all supported wall types have v_wall = 0
    v_f[0, :] = 0.0
    v_f[-1, :] = 0.0
    return u_f, v_f
