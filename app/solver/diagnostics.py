import numpy as np

from . import poisson_mg


def vorticity(u, v, dx, dy, bcs) -> np.ndarray:
    u_pad, v_pad = bcs.pad_velocity(u, v)
    dvdx = (v_pad[1:-1, 2:] - v_pad[1:-1, :-2]) / (2.0 * dx)
    dudy = (u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]) / (2.0 * dy)
    return dvdx - dudy


def streamfunction(u, v, dx, dy, U_ref=1.0, mg_tol=1e-6, mg_max=80) -> np.ndarray:
    """Diagnostic streamfunction ψ from the 2D velocity field.

    Solves ∇²ψ = -ω. Substituting ψ = ψ' + U_ref·y (the uniform-flow part):
      ∇²ψ' = -ω (since ∇²(U_ref·y) = 0) with ψ' = 0 on all boundaries.
    The Dirichlet-everywhere problem is solved with poisson_mg.BC_PSI.
    """
    ny, nx = u.shape
    u_pad = np.pad(u, 1, mode="edge")
    v_pad = np.pad(v, 1, mode="edge")
    omega = (
        (v_pad[1:-1, 2:] - v_pad[1:-1, :-2]) / (2.0 * dx)
        - (u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]) / (2.0 * dy)
    )

    psi_prime, _, _ = poisson_mg.solve(
        -omega, dx, dy, tol=mg_tol, max_cycles=mg_max, bc_sides=poisson_mg.BC_PSI
    )
    y = (np.arange(ny) + 0.5) * dy
    return psi_prime + U_ref * y[:, None]
