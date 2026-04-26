"""GPU vs NumPy parity for the four CuPy RawKernels in `_cupy_kernels.py`.

Skipped on machines without CuPy. Compares each GPU kernel against the
pure-NumPy reference kernels (`_np_*` in `_jit.py`) at fp64 to ~1e-10.
Mirrors the structure of `tests/test_jit_parity.py`.
"""
import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from app.solver._jit import (
    _np_apply_A, _np_rb_gs, _np_predictor, _np_rc_faces_interior,
)
from app.solver._cupy_kernels import (
    gpu_apply_A, gpu_rb_gs, gpu_predictor, gpu_rc_faces_interior,
)

BC_PCORR = (1, 0, 1, 1)
BC_PSI   = (0, 0, 0, 0)


def _rand(shape, seed):
    return np.random.RandomState(seed).standard_normal(shape)


def test_apply_A_parity_pcorr_gpu():
    p = _rand((32, 24), 0)
    dx, dy = 0.1, 0.08
    ref = _np_apply_A(p, dx, dy, *BC_PCORR)
    got = cupy.asnumpy(gpu_apply_A(cupy.asarray(p), dx, dy, *BC_PCORR))
    assert np.max(np.abs(ref - got)) < 1e-10


def test_apply_A_parity_psi_gpu():
    p = _rand((16, 32), 1)
    dx, dy = 0.05, 0.1
    ref = _np_apply_A(p, dx, dy, *BC_PSI)
    got = cupy.asnumpy(gpu_apply_A(cupy.asarray(p), dx, dy, *BC_PSI))
    assert np.max(np.abs(ref - got)) < 1e-10


def test_rb_gs_parity_gpu():
    p0 = _rand((32, 24), 2)
    rhs = _rand((32, 24), 3)
    dx, dy = 0.1, 0.08
    diag = np.full_like(p0, -2.0 / dx**2 - 2.0 / dy**2)
    ref = _np_rb_gs(p0, rhs, dx, dy, 3, diag, *BC_PCORR)
    got = cupy.asnumpy(gpu_rb_gs(
        cupy.asarray(p0), cupy.asarray(rhs), dx, dy, 3,
        cupy.asarray(diag), *BC_PCORR,
    ))
    assert np.max(np.abs(ref - got)) < 1e-10
    # Input must not be mutated by the GPU path
    assert np.all(p0 == _rand((32, 24), 2))


def test_predictor_parity_no_obstacle_gpu():
    ny, nx = 16, 24
    u = 1.0 + 0.2 * _rand((ny, nx), 10)
    v = 0.1 * _rand((ny, nx), 11)
    u_pad = np.pad(u, 1, mode="edge")
    v_pad = np.pad(v, 1, mode="edge")
    p_pad = np.pad(_rand((ny, nx), 12), 1, mode="edge")
    chi = np.zeros((ny, nx), dtype=np.bool_)
    dt, dx, dy, nu, eta = 0.01, 0.1, 0.08, 0.025, 1e-3
    ref = _np_predictor(u, v, u_pad, v_pad, p_pad, chi, dt, dx, dy, nu, eta, 0.0, 0.0)
    got = gpu_predictor(
        cupy.asarray(u), cupy.asarray(v),
        cupy.asarray(u_pad), cupy.asarray(v_pad), cupy.asarray(p_pad),
        cupy.asarray(chi),
        dt, dx, dy, nu, eta, 0.0, 0.0,
    )
    for r_arr, g_arr, name in zip(ref, got, ("u_star", "v_star", "a_P")):
        assert np.max(np.abs(r_arr - cupy.asnumpy(g_arr))) < 1e-10, f"{name} differs"


def test_predictor_parity_with_brinkman_gpu():
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
    got = gpu_predictor(
        cupy.asarray(u), cupy.asarray(v),
        cupy.asarray(u_pad), cupy.asarray(v_pad), cupy.asarray(p_pad),
        cupy.asarray(chi),
        dt, dx, dy, nu, eta, 0.1, 0.0,
    )
    for r_arr, g_arr, name in zip(ref, got, ("u_star", "v_star", "a_P")):
        assert np.max(np.abs(r_arr - cupy.asnumpy(g_arr))) < 1e-10, f"{name} differs"


def test_rc_faces_parity_gpu():
    ny, nx = 16, 24
    u = 1.0 + 0.2 * _rand((ny, nx), 20)
    v = 0.1 * _rand((ny, nx), 21)
    p_pad = np.pad(_rand((ny, nx), 22), 1, mode="edge")
    a_P = 1.0 + 0.5 * np.abs(_rand((ny, nx), 23))
    dt, dx, dy = 0.01, 0.1, 0.08

    u_f_ref = np.zeros((ny, nx + 1))
    v_f_ref = np.zeros((ny + 1, nx))
    _np_rc_faces_interior(u, v, p_pad, a_P, dt, dx, dy, u_f_ref, v_f_ref)

    u_f_gpu = cupy.zeros((ny, nx + 1))
    v_f_gpu = cupy.zeros((ny + 1, nx))
    gpu_rc_faces_interior(
        cupy.asarray(u), cupy.asarray(v),
        cupy.asarray(p_pad), cupy.asarray(a_P),
        dt, dx, dy, u_f_gpu, v_f_gpu,
    )

    assert np.max(np.abs(u_f_ref[:, 1:-1] - cupy.asnumpy(u_f_gpu)[:, 1:-1])) < 1e-10
    assert np.max(np.abs(v_f_ref[1:-1, :] - cupy.asnumpy(v_f_gpu)[1:-1, :])) < 1e-10


def test_full_step_parity_gpu(monkeypatch):
    """One Solver._step on CPU vs GPU produces identical u, v, p within 1e-8.

    Slightly looser than the kernel-level 1e-10 because intermediate fields
    accumulate small rounding differences across the full predictor +
    projection pipeline (different reduction order, FMA, etc.).
    """
    import importlib
    monkeypatch.setenv("CFD_BACKEND", "numpy")

    # Force re-import of solver chain under numpy backend
    for mod in (
        "app.solver._xp",
        "app.solver._jit",
        "app.solver.bc",
        "app.solver.operators",
        "app.solver.predictor",
        "app.solver.projection",
        "app.solver.poisson_mg",
        "app.solver.solver",
    ):
        if mod in importlib.sys.modules:
            importlib.reload(importlib.sys.modules[mod])

    from app.solver.config import SolverConfig
    from app.solver.mesh import build_mesh, rasterize_disk
    from app.solver.solver import Solver

    mesh = build_mesh(4.0, 2.0, 32, 16)
    chi = rasterize_disk(1.0, 1.0, 0.3, mesh)
    bcs = {
        "inlet":  {"type": "inlet_velocity", "speed": 1.0, "angle_deg": 0.0},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=40.0, max_iters=1, log_interval=10**9)
    s_cpu = Solver(mesh, chi, bcs, cfg)
    r_cpu = s_cpu.solve()

    # Re-import everything under cupy backend
    monkeypatch.setenv("CFD_BACKEND", "cupy")
    for mod in (
        "app.solver._xp",
        "app.solver._cupy_kernels",
        "app.solver._jit",
        "app.solver.bc",
        "app.solver.operators",
        "app.solver.predictor",
        "app.solver.projection",
        "app.solver.poisson_mg",
        "app.solver.solver",
    ):
        if mod in importlib.sys.modules:
            importlib.reload(importlib.sys.modules[mod])

    from app.solver.solver import Solver as SolverGPU
    s_gpu = SolverGPU(mesh, chi, bcs, cfg)
    r_gpu = s_gpu.solve()

    assert np.max(np.abs(r_cpu.u - r_gpu.u)) < 1e-8
    assert np.max(np.abs(r_cpu.v - r_gpu.v)) < 1e-8
    assert np.max(np.abs(r_cpu.p - r_gpu.p)) < 1e-8
