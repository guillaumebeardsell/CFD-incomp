"""Verify grad/div/Laplacian on a manufactured field."""
import numpy as np

from app.solver.bc import Boundaries
from app.solver.operators import gradient, laplacian, divergence_from_faces


def _bcs():
    return Boundaries({
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    })


def test_gradient_of_linear_field_is_constant():
    nx, ny = 64, 48
    dx = 1.0 / nx; dy = 1.0 / ny
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    f = 3.0 * X[None, :] + 5.0 * Y[:, None]
    pad = np.pad(f, 1, mode="edge")
    gx, gy = gradient(pad, dx, dy)
    # Interior cells (away from edge-padding boundary) should give exact 3, 5.
    assert np.allclose(gx[:, 1:-1], 3.0, atol=1e-10)
    assert np.allclose(gy[1:-1, :], 5.0, atol=1e-10)


def test_laplacian_of_sin_matches_analytic():
    nx, ny = 64, 64
    Lx = 2.0; Ly = 2.0
    dx = Lx / nx; dy = Ly / ny
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    kx = np.pi / Lx; ky = np.pi / Ly
    f = np.sin(kx * X[None, :]) * np.sin(ky * Y[:, None])
    # For a sine field, pad with odd-reflection ghosts so stencil is clean.
    pad = np.empty((ny + 2, nx + 2))
    pad[1:-1, 1:-1] = f
    pad[:, 0] = pad[:, 1]; pad[:, -1] = pad[:, -2]
    pad[0, :] = pad[1, :]; pad[-1, :] = pad[-2, :]
    lap = laplacian(pad, dx, dy)
    lap_exact = -(kx ** 2 + ky ** 2) * f
    # Compare interior only — boundaries have edge-pad artifact.
    err = np.max(np.abs(lap[5:-5, 5:-5] - lap_exact[5:-5, 5:-5]))
    # Second-order accurate: expect ~ (pi/L/n)^2 * field
    assert err < 0.02


def test_divergence_from_faces_on_constant_field():
    nx, ny = 32, 32
    dx = 1.0 / nx; dy = 1.0 / ny
    u_face_e = np.full((ny, nx + 1), 1.0)
    v_face_n = np.zeros((ny + 1, nx))
    div = divergence_from_faces(u_face_e, v_face_n, dx, dy)
    assert np.allclose(div, 0.0, atol=1e-12)
