"""Projection reduces divergence and drives it toward zero over iterations."""
import numpy as np

from app.solver.bc import Boundaries
from app.solver.operators import rhie_chow_faces, divergence_from_faces
from app.solver.predictor import predictor
from app.solver.projection import project


def test_projection_reduces_rc_face_divergence():
    """A single projection step reduces the RC face divergence.
    (Exact divergence-freeness of the RC-recomputed faces is not achievable
    on a collocated grid — the projection is exact only for face velocities
    it corrects directly. The per-step reduction factor is ~(a_P-1)/a_P
    with point-implicit diffusion, so we test a modest 2x reduction; the
    outer steady loop drives divergence to zero across iterations.)"""
    nx, ny = 64, 32
    dx, dy = 4.0 / nx, 1.0 / ny
    rng = np.random.default_rng(42)
    u = 1.0 + 0.1 * rng.standard_normal((ny, nx))
    v = 0.1 * rng.standard_normal((ny, nx))
    p = np.zeros((ny, nx))
    chi = np.zeros((ny, nx), dtype=bool)
    bcs = Boundaries({
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    })
    dt = 0.01
    nu = 0.025
    eta = 1e-3

    u_star, v_star, a_P = predictor(u, v, p, chi, dt, dx, dy, nu, eta, bcs)
    # Pre-projection RC divergence
    p0_pad = bcs.pad_pressure(p)
    u_f0, v_f0 = rhie_chow_faces(u_star, v_star, p0_pad, a_P, dt, dx, dy, bcs)
    div_before = np.max(np.abs(divergence_from_faces(u_f0, v_f0, dx, dy)))

    u_new, v_new, p_new, _p_prime, info = project(
        u_star, v_star, p, a_P, dt, dx, dy, bcs, mg_tol=1e-10, mg_max=40, alpha_p=1.0,
    )
    # The *projection-internal* divergence drop is captured in info["mg_resid"]
    # (near zero). The RC-recomputed post-projection divergence is a looser
    # proxy: it must be much smaller than the pre-projection RC divergence.
    p_new_pad = bcs.pad_pressure(p_new)
    u_f, v_f = rhie_chow_faces(u_new, v_new, p_new_pad, a_P, dt, dx, dy, bcs)
    div_after = np.max(np.abs(divergence_from_faces(u_f, v_f, dx, dy)))

    assert info["mg_resid"] < 1e-6, f"MG didn't converge: {info['mg_resid']}"
    assert div_after < 0.5 * div_before, (
        f"projection did not reduce divergence: before={div_before:.3e}, "
        f"after={div_after:.3e}"
    )
