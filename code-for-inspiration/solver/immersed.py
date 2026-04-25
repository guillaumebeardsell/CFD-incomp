"""Ghost-cell immersed boundary.

Each solid cell adjacent to the fluid (the "ghost band") is paired with an
image point reflected across the true polygon boundary. Fluid state at the
image point is interpolated bilinearly from its 4 surrounding cell centers.
The wall BC is then enforced by constructing the ghost primitive state such
that the face-average between image and ghost satisfies the BC (slip: reflect
normal velocity; no-slip: reverse full velocity; isothermal: T_g = 2*T_w - T_im).

Fluxes on fluid/ghost faces are then computed by the ordinary Roe solver;
the reflection guarantees the flux reduces to the wall flux at the polygon.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def build_ghost_info(polygon_xy, mesh, solid_mask, band_width=2):
    """Precompute per-ghost-cell image-point stencils.

    Returns dict of aligned arrays (length N = number of valid ghost cells), or
    None if no ghosts exist. Invalid ghosts (image stencil hits a solid/out-of-
    domain cell) are dropped; those cells fall back to the outer initial state.
    """
    if polygon_xy is None or len(polygon_xy) < 3 or not solid_mask.any():
        return None

    ny, nx = mesh["ny"], mesh["nx"]
    dx, dy = mesh["dx"], mesh["dy"]

    fluid = ~solid_mask
    ghost_band = solid_mask & binary_dilation(fluid, iterations=band_width)
    jj, ii = np.where(ghost_band)
    if len(jj) == 0:
        return None

    poly = Polygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    boundary = poly.boundary

    N = len(jj)
    sten_j = np.zeros((N, 4), dtype=np.int32)
    sten_i = np.zeros((N, 4), dtype=np.int32)
    weights = np.zeros((N, 4), dtype=np.float64)
    nxh = np.zeros(N)
    nyh = np.zeros(N)
    valid = np.zeros(N, dtype=bool)

    # 8-connected neighbour offsets, ordered by distance so the nearest-fluid
    # fallback picks the closest one first.
    nbr_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def nearest_fluid(j, i):
        for dj, di in nbr_offsets:
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and not solid_mask[nj, ni]:
                return nj, ni
        return None

    for k in range(N):
        j, i = int(jj[k]), int(ii[k])
        xg = (i + 0.5) * dx
        yg = (j + 0.5) * dy
        near, _ = nearest_points(boundary, Point(xg, yg))
        xb, yb = near.x, near.y
        d = np.hypot(xb - xg, yb - yg)

        # Outward normal from ghost (inside body) to boundary.
        if d >= 1e-12:
            nxh[k] = (xb - xg) / d
            nyh[k] = (yb - yg) / d
        # else: zero normal; BC reflection will still no-op sensibly and the
        # fallback stencil below keeps the cell from holding freestream.

        bilinear_ok = False
        if d >= 1e-12:
            xi = 2.0 * xb - xg
            yi = 2.0 * yb - yg
            u = xi / dx - 0.5
            v = yi / dy - 0.5
            i0 = int(np.floor(u))
            j0 = int(np.floor(v))
            fx = u - i0
            fy = v - j0
            si = np.array([i0, i0 + 1, i0, i0 + 1], dtype=np.int32)
            sj = np.array([j0, j0, j0 + 1, j0 + 1], dtype=np.int32)
            in_bounds = (
                (si >= 0).all() and (si < nx).all() and (sj >= 0).all() and (sj < ny).all()
            )
            if in_bounds and not solid_mask[sj, si].any():
                sten_i[k] = si
                sten_j[k] = sj
                weights[k] = [
                    (1 - fx) * (1 - fy), fx * (1 - fy),
                    (1 - fx) * fy, fx * fy,
                ]
                valid[k] = True
                bilinear_ok = True

        if not bilinear_ok:
            # Fallback: use nearest fluid neighbour as the image state (weight
            # 1). Geometrically coarse — the reflected image isn't actually at
            # that cell centre — but guarantees the ghost holds fluid-side
            # state rather than the stale freestream IC.
            nb = nearest_fluid(j, i)
            if nb is not None:
                nj, ni = nb
                sten_j[k] = [nj, nj, nj, nj]
                sten_i[k] = [ni, ni, ni, ni]
                weights[k] = [1.0, 0.0, 0.0, 0.0]
                valid[k] = True

    if not valid.any():
        return None

    return {
        "j": jj[valid].astype(np.int32),
        "i": ii[valid].astype(np.int32),
        "sten_j": sten_j[valid],
        "sten_i": sten_i[valid],
        "weights": weights[valid],
        "nx_hat": nxh[valid],
        "ny_hat": nyh[valid],
    }


def fill_ghosts(W, ghost_info, obstacle_bc, R):
    """Overwrite primitive state at ghost cells in-place.

    W has shape (4, ny+4, nx+4) with a 2-cell padding; interior cell (j, i)
    lives at padded index (j+2, i+2). Stencil indices in `ghost_info` refer
    to interior cells, so we add 2 to each before indexing.
    """
    if ghost_info is None:
        return
    j = ghost_info["j"]; i = ghost_info["i"]
    sj = ghost_info["sten_j"] + 2
    si = ghost_info["sten_i"] + 2
    w = ghost_info["weights"]
    nxh = ghost_info["nx_hat"]
    nyh = ghost_info["ny_hat"]

    # Bilinear interpolation of each primitive at the image point (N,).
    rho_im = (W[0, sj, si] * w).sum(axis=1)
    u_im   = (W[1, sj, si] * w).sum(axis=1)
    v_im   = (W[2, sj, si] * w).sum(axis=1)
    p_im   = (W[3, sj, si] * w).sum(axis=1)

    vn = u_im * nxh + v_im * nyh
    bc_type = obstacle_bc.get("type", "slip_wall")

    if bc_type == "no_slip_wall":
        u_g = -u_im
        v_g = -v_im
        p_g = p_im
        T_wall = obstacle_bc.get("T_wall")
        if T_wall is not None:
            T_im = p_im / np.maximum(rho_im * R, 1e-30)
            T_g = np.maximum(2.0 * T_wall - T_im, 0.5 * T_wall)  # clamp for robustness
            rho_g = p_g / (R * T_g)
        else:  # adiabatic: zero-gradient in T
            rho_g = rho_im
    else:  # slip_wall / symmetry: reflect normal velocity only
        u_g = u_im - 2.0 * vn * nxh
        v_g = v_im - 2.0 * vn * nyh
        rho_g = rho_im
        p_g = p_im

    pj = j + 2
    pi = i + 2
    W[0, pj, pi] = np.maximum(rho_g, 1e-6)
    W[1, pj, pi] = u_g
    W[2, pj, pi] = v_g
    W[3, pj, pi] = np.maximum(p_g, 1.0)
