"""Brinkman penalization: velocity inside solid must shrink as eta -> 0."""
import numpy as np

from app.solver import Solver, SolverConfig
from app.solver.mesh import build_mesh


def _run(eta, max_iters=800):
    mesh = build_mesh(4.0, 1.0, 64, 16)
    chi = np.zeros((16, 64), dtype=bool)
    chi[5:11, 22:28] = True  # small solid block
    bcs = {
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=40.0, eta=eta,
                       max_iters=max_iters, res_drop=1e-6,
                       log_interval=10_000)
    r = Solver(mesh, chi, bcs, cfg).solve()
    return r, chi


def test_solid_velocity_scales_with_eta():
    r1, chi1 = _run(eta=1e-2)
    r2, chi2 = _run(eta=1e-4)
    u_solid_1 = float(np.max(np.abs(r1.u[chi1])))
    u_solid_2 = float(np.max(np.abs(r2.u[chi2])))
    # eta drops by 100x -> solid velocity should drop by at least 10x
    assert u_solid_2 < u_solid_1 / 10.0, (
        f"u_solid: eta=1e-2 -> {u_solid_1:.3e}, eta=1e-4 -> {u_solid_2:.3e}"
    )
    # Absolute bound: solid velocity smaller than freestream by a lot
    assert u_solid_2 < 0.05, f"u_solid too large: {u_solid_2}"
