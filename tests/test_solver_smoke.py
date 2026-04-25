"""Smoke test: empty tunnel preserves freestream; cylinder case runs without crashing."""
import numpy as np

from app.solver import Solver, SolverConfig
from app.solver.mesh import build_mesh, rasterize_disk


def test_empty_tunnel_preserves_freestream():
    mesh = build_mesh(4.0, 1.0, 64, 16)
    chi = np.zeros((16, 64), dtype=bool)
    bcs = {
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=40.0, max_iters=100, log_interval=10_000)
    r = Solver(mesh, chi, bcs, cfg).solve()
    assert np.allclose(r.u, 1.0, atol=1e-8)
    assert np.allclose(r.v, 0.0, atol=1e-8)
    assert np.max(np.abs(r.p)) < 1e-6


def test_cylinder_runs_and_stabilizes():
    mesh = build_mesh(8.0, 4.0, 80, 40)
    chi = rasterize_disk(2.0, 2.0, 0.5, mesh)
    bcs = {
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=40.0, eta=1e-3,
                       max_iters=400, res_drop=1e-3, log_interval=10_000)
    r = Solver(mesh, chi, bcs, cfg).solve()
    assert np.isfinite(r.u).all() and np.isfinite(r.v).all() and np.isfinite(r.p).all()
    # Wake region: downstream of cylinder, centerline velocity should be less than 1
    iy = mesh["ny"] // 2
    ix_wake = 28  # a bit downstream of cylinder
    assert r.u[iy, ix_wake] < 1.0
    # Pressure near front of cylinder should exceed pressure near wake
    p_front = r.p[iy, 15]
    p_back = r.p[iy, 25]
    assert p_front > p_back
