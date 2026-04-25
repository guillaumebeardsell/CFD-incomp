"""Canonical CFD cases (qualitative / quantitative sanity checks).

Uses modest grids + iteration counts to keep CI fast; acceptance thresholds
are intentionally loose because v1 uses a 2nd-order FV scheme with a
staircase immersed boundary.
"""
import numpy as np

from solver import mesh as mesh_mod
from solver.solver import Solver


def test_freestream_preservation_inviscid():
    """Empty tunnel + uniform inlet should remain uniform to machine precision."""
    m = mesh_mod.build_mesh(1.0, 0.5, 40, 20)
    solid = np.zeros((20, 40), dtype=bool)
    bcs = {
        "inlet":  {"type": "inlet_subsonic", "mach": 0.3, "p": 101325.0, "T": 300.0},
        "outlet": {"type": "outlet_subsonic", "p": 101325.0},
        "top":    {"type": "slip_wall"},
        "bottom": {"type": "slip_wall"},
    }
    solver = Solver(m, solid, bcs, max_iters=50, res_drop=1e-14)
    result = solver.solve()
    # Density, u, v should remain close to freestream; pressure is constant.
    rho0 = 101325.0 / (287.05 * 300.0)
    c0 = np.sqrt(1.4 * 101325.0 / rho0)
    u0 = 0.3 * c0
    np.testing.assert_allclose(result.W[0], rho0, rtol=1e-3)
    np.testing.assert_allclose(result.W[1], u0, rtol=1e-3)
    assert abs(result.W[2]).max() < 1e-2 * u0
    np.testing.assert_allclose(result.W[3], 101325.0, rtol=1e-3)


def test_subsonic_cylinder_qualitative():
    """Flow past a cylinder should accelerate at the shoulders and create a forward stagnation region."""
    m = mesh_mod.build_mesh(width=2.0, height=1.0, nx=120, ny=60)
    cx, cy, r = 0.6, 0.5, 0.12
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    poly = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in theta]
    solid = mesh_mod.rasterize_polygon(poly, m)
    bcs = {
        "inlet":  {"type": "inlet_subsonic", "mach": 0.3, "p": 101325.0, "T": 300.0},
        "outlet": {"type": "outlet_subsonic", "p": 101325.0},
        "top":    {"type": "slip_wall"},
        "bottom": {"type": "slip_wall"},
    }
    solver = Solver(m, solid, bcs, max_iters=400, res_drop=1e-8, polygon_xy=poly)
    result = solver.solve()

    # Grid-to-world for locating shoulders / stagnation.
    dx, dy = m["dx"], m["dy"]
    ix_front = int((cx - r - 2 * dx) / dx)   # a few cells upstream of the cylinder
    iy_mid   = int(cy / dy)
    jx_shoulder = int(cx / dx)
    iy_top_shoulder = int((cy + r + 2 * dy) / dy)

    rho0 = 101325.0 / (287.05 * 300.0)
    c0 = np.sqrt(1.4 * 101325.0 / rho0)
    u0 = 0.3 * c0

    # Stagnation: u << u_inf just upstream of cylinder along centreline.
    u_front = result.W[1, iy_mid, ix_front]
    assert u_front < 0.65 * u0, f"expected stagnation upstream, got u/u_inf = {u_front / u0:.2f}"

    # Shoulder: flow accelerated above cylinder.
    u_shoulder = result.W[1, iy_top_shoulder, jx_shoulder]
    assert u_shoulder > 1.1 * u0, f"expected acceleration at shoulder, got u/u_inf = {u_shoulder / u0:.2f}"

    # Symmetry: Mach field should be roughly symmetric about cy.
    m_top = result.mach[iy_top_shoulder, jx_shoulder]
    iy_bot_shoulder = int((cy - r - 2 * dy) / dy)
    m_bot = result.mach[iy_bot_shoulder, jx_shoulder]
    assert abs(m_top - m_bot) / max(m_top, m_bot) < 0.1, (
        f"top/bottom shoulder symmetry broken: {m_top:.3f} vs {m_bot:.3f}"
    )
