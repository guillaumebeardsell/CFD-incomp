import numpy as np

from solver import state as state_mod
from solver.flux_inviscid import roe_flux_x, roe_flux_y, euler_flux_x


def test_state_roundtrip():
    W = np.array([[1.225], [50.0], [10.0], [101325.0]])
    U = state_mod.primitive_to_conservative(W)
    W2 = state_mod.conservative_to_primitive(U)
    np.testing.assert_allclose(W, W2, rtol=1e-10)


def test_roe_flux_uniform_state_matches_euler_flux():
    """With identical left/right states the Roe dissipation vanishes and F = F_L."""
    W = np.zeros((4, 3, 3))
    W[0] = 1.225
    W[1] = 50.0
    W[2] = 10.0
    W[3] = 101325.0
    F = roe_flux_x(W, W)
    F_exact = euler_flux_x(W)
    np.testing.assert_allclose(F, F_exact, rtol=1e-10, atol=1e-8)


def test_roe_flux_x_vs_y_symmetry():
    """Flow aligned with +y should give y-flux equal to x-flux of same flow rotated."""
    Wx = np.zeros((4, 3, 3))
    Wx[0] = 1.0; Wx[1] = 100.0; Wx[2] = 0.0; Wx[3] = 100000.0
    Wy = np.zeros((4, 3, 3))
    Wy[0] = 1.0; Wy[1] = 0.0;   Wy[2] = 100.0; Wy[3] = 100000.0
    Fx = roe_flux_x(Wx, Wx)
    Fy = roe_flux_y(Wy, Wy)
    # Mass flux and energy flux equal; momentum components swap.
    np.testing.assert_allclose(Fx[0], Fy[0], rtol=1e-10)
    np.testing.assert_allclose(Fx[1], Fy[2], rtol=1e-10)
    np.testing.assert_allclose(Fx[3], Fy[3], rtol=1e-10)
