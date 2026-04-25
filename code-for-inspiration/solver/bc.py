"""Boundary conditions applied via ghost cells.

Layout: state arrays are padded with 2 ghost layers on each side so they have
shape (4, ny+4, nx+4). The fluid interior lives at [:, 2:ny+2, 2:nx+2].
All BCs operate on primitive variables W = [rho, u, v, p].
"""
import numpy as np

from . import config


def _fill_from_primitive(Wp, rho, u, v, p, slc):
    Wp[0][slc] = rho
    Wp[1][slc] = u
    Wp[2][slc] = v
    Wp[3][slc] = p


def apply_bcs(Wp, bcs, gamma=config.GAMMA, R=config.R_GAS):
    """Fill ghost cells on all four tunnel boundaries.

    Wp : (4, ny+4, nx+4) primitive array. Mutated in place.
    bcs : dict with keys 'inlet', 'outlet', 'top', 'bottom'.
          Each value is a dict with at least a 'type' key.
    """
    ny_p, nx_p = Wp.shape[1], Wp.shape[2]
    # Interior slice for copying nearest-interior cell to the two ghost layers.

    # Left (inlet) --------------------------------------------------------
    bc = bcs["inlet"]
    if bc["type"] == "inlet_subsonic":
        rho = bc["p"] / (R * bc["T"])
        c = np.sqrt(gamma * bc["p"] / rho)
        u_inf = bc["mach"] * c
        # Extrapolate pressure from interior, fix rho, u, v.
        for k in range(2):
            p_ext = Wp[3, :, 2]  # first interior column
            Wp[0, :, k] = rho
            Wp[1, :, k] = u_inf
            Wp[2, :, k] = 0.0
            Wp[3, :, k] = p_ext
    elif bc["type"] == "inlet_supersonic":
        rho = bc["p"] / (R * bc["T"])
        c = np.sqrt(gamma * bc["p"] / rho)
        u_inf = bc["mach"] * c
        for k in range(2):
            Wp[0, :, k] = rho
            Wp[1, :, k] = u_inf
            Wp[2, :, k] = 0.0
            Wp[3, :, k] = bc["p"]
    elif bc["type"] == "slip_wall":
        # Mirror: flip u, keep rho, v, p.
        Wp[0, :, 0] = Wp[0, :, 3]; Wp[0, :, 1] = Wp[0, :, 2]
        Wp[1, :, 0] = -Wp[1, :, 3]; Wp[1, :, 1] = -Wp[1, :, 2]
        Wp[2, :, 0] = Wp[2, :, 3]; Wp[2, :, 1] = Wp[2, :, 2]
        Wp[3, :, 0] = Wp[3, :, 3]; Wp[3, :, 1] = Wp[3, :, 2]
    elif bc["type"] == "no_slip_wall":
        Wp[0, :, 0] = Wp[0, :, 3]; Wp[0, :, 1] = Wp[0, :, 2]
        Wp[1, :, 0] = -Wp[1, :, 3]; Wp[1, :, 1] = -Wp[1, :, 2]
        Wp[2, :, 0] = -Wp[2, :, 3]; Wp[2, :, 1] = -Wp[2, :, 2]
        Wp[3, :, 0] = Wp[3, :, 3]; Wp[3, :, 1] = Wp[3, :, 2]
    else:
        raise ValueError(f"unknown inlet BC type {bc['type']!r}")

    # Right (outlet) ------------------------------------------------------
    bc = bcs["outlet"]
    if bc["type"] == "outlet_subsonic":
        for k in range(2):
            col = nx_p - 1 - k
            Wp[0, :, col] = Wp[0, :, nx_p - 3]
            Wp[1, :, col] = Wp[1, :, nx_p - 3]
            Wp[2, :, col] = Wp[2, :, nx_p - 3]
            Wp[3, :, col] = bc["p"]
    elif bc["type"] == "outlet_supersonic":
        for k in range(2):
            col = nx_p - 1 - k
            src = nx_p - 3
            Wp[:, :, col] = Wp[:, :, src]
    elif bc["type"] == "slip_wall":
        Wp[0, :, -1] = Wp[0, :, -4]; Wp[0, :, -2] = Wp[0, :, -3]
        Wp[1, :, -1] = -Wp[1, :, -4]; Wp[1, :, -2] = -Wp[1, :, -3]
        Wp[2, :, -1] = Wp[2, :, -4]; Wp[2, :, -2] = Wp[2, :, -3]
        Wp[3, :, -1] = Wp[3, :, -4]; Wp[3, :, -2] = Wp[3, :, -3]
    elif bc["type"] == "no_slip_wall":
        Wp[0, :, -1] = Wp[0, :, -4]; Wp[0, :, -2] = Wp[0, :, -3]
        Wp[1, :, -1] = -Wp[1, :, -4]; Wp[1, :, -2] = -Wp[1, :, -3]
        Wp[2, :, -1] = -Wp[2, :, -4]; Wp[2, :, -2] = -Wp[2, :, -3]
        Wp[3, :, -1] = Wp[3, :, -4]; Wp[3, :, -2] = Wp[3, :, -3]
    else:
        raise ValueError(f"unknown outlet BC type {bc['type']!r}")

    # Bottom and top (y=0 and y=H) ---------------------------------------
    for side in ("bottom", "top"):
        bc = bcs[side]
        if side == "bottom":
            g0, g1, i0, i1 = 0, 1, 3, 2  # ghost rows 0,1 reflect interior rows 3,2
        else:
            g0, g1, i0, i1 = ny_p - 1, ny_p - 2, ny_p - 4, ny_p - 3

        if bc["type"] == "slip_wall":
            Wp[0, g0, :] = Wp[0, i0, :]; Wp[0, g1, :] = Wp[0, i1, :]
            Wp[1, g0, :] = Wp[1, i0, :]; Wp[1, g1, :] = Wp[1, i1, :]
            Wp[2, g0, :] = -Wp[2, i0, :]; Wp[2, g1, :] = -Wp[2, i1, :]
            Wp[3, g0, :] = Wp[3, i0, :]; Wp[3, g1, :] = Wp[3, i1, :]
        elif bc["type"] == "no_slip_wall":
            Wp[0, g0, :] = Wp[0, i0, :]; Wp[0, g1, :] = Wp[0, i1, :]
            Wp[1, g0, :] = -Wp[1, i0, :]; Wp[1, g1, :] = -Wp[1, i1, :]
            Wp[2, g0, :] = -Wp[2, i0, :]; Wp[2, g1, :] = -Wp[2, i1, :]
            Wp[3, g0, :] = Wp[3, i0, :]; Wp[3, g1, :] = Wp[3, i1, :]
        elif bc["type"] == "symmetry":
            Wp[0, g0, :] = Wp[0, i0, :]; Wp[0, g1, :] = Wp[0, i1, :]
            Wp[1, g0, :] = Wp[1, i0, :]; Wp[1, g1, :] = Wp[1, i1, :]
            Wp[2, g0, :] = -Wp[2, i0, :]; Wp[2, g1, :] = -Wp[2, i1, :]
            Wp[3, g0, :] = Wp[3, i0, :]; Wp[3, g1, :] = Wp[3, i1, :]
        else:
            raise ValueError(f"unknown {side} BC type {bc['type']!r}")
