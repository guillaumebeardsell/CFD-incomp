"""JIT vs NumPy parity tests for the hot kernels.

Every @njit kernel in app/solver/_jit.py has a pure-NumPy reference
implementation that we verify matches element-wise. Tolerance is 1e-9 for
single-kernel parity and 1e-8 for full MG / full-step parity — loosened
from 1e-12 / 1e-10 to accommodate FMA and reassociation drift from
fastmath=True on the Numba path.
"""
import numpy as np
import pytest

from app.solver import _jit
from app.solver._jit import (
    _np_apply_A, _np_rb_gs, _np_predictor, _np_rc_faces_interior,
    _jit_apply_A, _jit_rb_gs, _jit_predictor, _jit_rc_faces_interior,
    USE_JIT,
)

pytestmark = pytest.mark.skipif(not USE_JIT, reason="numba not installed")

# DIRICHLET=0, NEUMANN=1 (matches module constants)
BC_PCORR = (1, 0, 1, 1)   # west, east, south, north
BC_PSI   = (0, 0, 0, 0)


def _rand(shape, seed):
    return np.random.RandomState(seed).standard_normal(shape)


def test_apply_A_parity_pcorr():
    p = _rand((32, 24), 0)
    dx, dy = 0.1, 0.08
    ref = _np_apply_A(p, dx, dy, *BC_PCORR)
    got = _jit_apply_A(p, dx, dy, *BC_PCORR)
    assert np.max(np.abs(ref - got)) < 1e-9


def test_apply_A_parity_psi():
    p = _rand((16, 32), 1)
    dx, dy = 0.05, 0.1
    ref = _np_apply_A(p, dx, dy, *BC_PSI)
    got = _jit_apply_A(p, dx, dy, *BC_PSI)
    assert np.max(np.abs(ref - got)) < 1e-9


def test_rb_gs_parity():
    p0 = _rand((32, 24), 2)
    rhs = _rand((32, 24), 3)
    dx, dy = 0.1, 0.08
    diag = np.full_like(p0, -2.0 / dx**2 - 2.0 / dy**2)
    ref = _np_rb_gs(p0, rhs, dx, dy, 3, diag, *BC_PCORR)
    got = _jit_rb_gs(p0, rhs, dx, dy, 3, diag, *BC_PCORR)
    assert np.max(np.abs(ref - got)) < 1e-9
    # Input must not be mutated by either path
    assert np.all(p0 == _rand((32, 24), 2))


def test_poisson_solve_parity():
    """Full MG solve with JIT vs NumPy produces identical residual history."""
    import os
    from app.solver import poisson_mg

    ny, nx = 32, 32
    dx, dy = 1.0 / nx, 1.0 / ny
    # Manufactured Poisson rhs
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    rhs = -2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    p_jit, n_jit, r_jit = poisson_mg.solve(rhs, dx, dy, tol=1e-8, max_cycles=20)

    # Force fallback via monkeypatch
    orig_apply, orig_rb = _jit.apply_A_kernel, _jit.rb_gs_kernel
    _jit.apply_A_kernel = _jit._np_apply_A
    _jit.rb_gs_kernel = _jit._np_rb_gs
    # Re-import poisson_mg's cached references
    import importlib
    importlib.reload(poisson_mg)
    try:
        p_np, n_np, r_np = poisson_mg.solve(rhs, dx, dy, tol=1e-8, max_cycles=20)
    finally:
        _jit.apply_A_kernel = orig_apply
        _jit.rb_gs_kernel = orig_rb
        importlib.reload(poisson_mg)

    assert n_jit == n_np, f"cycle counts differ: jit={n_jit} np={n_np}"
    assert np.max(np.abs(p_jit - p_np)) < 1e-8


def test_predictor_parity_no_obstacle():
    ny, nx = 16, 24
    u = 1.0 + 0.2 * _rand((ny, nx), 10)
    v = 0.1 * _rand((ny, nx), 11)
    # Build padded arrays manually mirroring bc.py convention
    u_pad = np.pad(u, 1, mode="edge")
    v_pad = np.pad(v, 1, mode="edge")
    p_pad = np.pad(_rand((ny, nx), 12), 1, mode="edge")
    chi = np.zeros((ny, nx), dtype=np.bool_)
    dt, dx, dy, nu, eta = 0.01, 0.1, 0.08, 0.025, 1e-3
    ref = _np_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta, 0.0, 0.0)
    got = _jit_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta, 0.0, 0.0)
    for r_arr, g_arr, name in zip(ref, got, ("u_star", "v_star", "a_P")):
        assert np.max(np.abs(r_arr - g_arr)) < 1e-9, f"{name} differs"


def test_predictor_parity_with_brinkman():
    ny, nx = 16, 24
    u = 1.0 + 0.2 * _rand((ny, nx), 13)
    v = 0.1 * _rand((ny, nx), 14)
    u_pad = np.pad(u, 1, mode="edge")
    v_pad = np.pad(v, 1, mode="edge")
    p_pad = np.pad(_rand((ny, nx), 15), 1, mode="edge")
    chi = np.zeros((ny, nx), dtype=np.bool_)
    chi[6:10, 8:14] = True
    dt, dx, dy, nu, eta = 0.005, 0.1, 0.08, 0.025, 1e-3
    ref = _np_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta, 0.1, 0.0)
    got = _jit_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta, 0.1, 0.0)
    for r_arr, g_arr, name in zip(ref, got, ("u_star", "v_star", "a_P")):
        assert np.max(np.abs(r_arr - g_arr)) < 1e-9, f"{name} differs"


def test_rc_faces_parity():
    ny, nx = 16, 24
    u = 1.0 + 0.2 * _rand((ny, nx), 20)
    v = 0.1 * _rand((ny, nx), 21)
    p_pad = np.pad(_rand((ny, nx), 22), 1, mode="edge")
    a_P = 1.0 + 0.5 * np.abs(_rand((ny, nx), 23))
    dt, dx, dy = 0.01, 0.1, 0.08

    u_f_ref = np.zeros((ny, nx + 1))
    v_f_ref = np.zeros((ny + 1, nx))
    _np_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f_ref, v_f_ref)

    u_f_jit = np.zeros((ny, nx + 1))
    v_f_jit = np.zeros((ny + 1, nx))
    _jit_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f_jit, v_f_jit)

    # Only interior faces are written; check those slices
    assert np.max(np.abs(u_f_ref[:, 1:-1] - u_f_jit[:, 1:-1])) < 1e-9
    assert np.max(np.abs(v_f_ref[1:-1, :] - v_f_jit[1:-1, :])) < 1e-9


def test_full_step_parity():
    """One Solver._step with JIT on vs off produces identical u, v, p."""
    import importlib
    from app.solver import poisson_mg, predictor as predmod, operators
    from app.solver.solver import Solver
    from app.solver.config import SolverConfig
    from app.solver.mesh import build_mesh, rasterize_disk

    mesh = build_mesh(4.0, 2.0, 32, 16)
    chi = rasterize_disk(1.0, 1.0, 0.3, mesh)
    bcs = {
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=40.0, max_iters=1, log_interval=10**9)

    # JIT pass
    s1 = Solver(mesh, chi, bcs, cfg)
    r1 = s1.solve()

    # NumPy pass via monkeypatch
    orig = (_jit.apply_A_kernel, _jit.rb_gs_kernel,
            _jit.predictor_kernel, _jit.rc_faces_interior_kernel)
    _jit.apply_A_kernel = _jit._np_apply_A
    _jit.rb_gs_kernel = _jit._np_rb_gs
    _jit.predictor_kernel = _jit._np_predictor
    _jit.rc_faces_interior_kernel = _jit._np_rc_faces_interior
    importlib.reload(poisson_mg)
    importlib.reload(predmod)
    importlib.reload(operators)
    try:
        # Need a fresh Solver module import because solver.py references
        # predictor/project by symbol
        from app.solver import solver as solver_mod
        importlib.reload(solver_mod)
        s2 = solver_mod.Solver(mesh, chi, bcs, cfg)
        r2 = s2.solve()
    finally:
        (_jit.apply_A_kernel, _jit.rb_gs_kernel,
         _jit.predictor_kernel, _jit.rc_faces_interior_kernel) = orig
        importlib.reload(poisson_mg)
        importlib.reload(predmod)
        importlib.reload(operators)

    assert np.max(np.abs(r1.u - r2.u)) < 1e-8
    assert np.max(np.abs(r1.v - r2.v)) < 1e-8
    assert np.max(np.abs(r1.p - r2.p)) < 1e-8
