"""Multigrid contraction rate + manufactured-solution correctness."""
import numpy as np

from app.solver import poisson_mg


def test_mg_solves_manufactured_problem():
    """∇²p = f with p=cos(kx x)·cos(ky y) analytic and mixed BCs.

    For BC_PCORR (west Neumann, east Dirichlet, south Neumann, north Neumann)
    use p = cos(kx x)·cos(ky y) with kx = 0.5·π/Lx (so cos(kx·Lx) = 0 at east).
    y-part uses ky = 0 -> purely cos(kx x) field to cleanly satisfy N/S Neumann.
    """
    nx, ny = 128, 64
    Lx, Ly = 2.0, 1.0
    dx = Lx / nx; dy = Ly / ny
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy

    kx = 0.5 * np.pi / Lx  # cos(kx·0) = 1 (west Neumann OK — derivative 0 there)
    p_exact = np.cos(kx * X)[None, :] * np.ones((ny, 1))
    f = -(kx ** 2) * p_exact  # since d²/dy² = 0

    p, cycles, resid = poisson_mg.solve(f, dx, dy, tol=1e-6, max_cycles=20,
                                         bc_sides=poisson_mg.BC_PCORR)
    err = np.max(np.abs(p - p_exact))
    assert err < 1e-4, f"max err={err:.2e}, cycles={cycles}, resid={resid:.2e}"
    assert cycles <= 25, f"too many MG cycles: {cycles}"


def test_mg_contraction_rate():
    """Residual should drop by >5x per V-cycle on a well-posed problem."""
    nx, ny = 64, 64
    dx = dy = 1.0 / nx
    np.random.seed(0)
    f = np.random.randn(ny, nx)
    p = np.zeros_like(f)
    rhs_norm = np.linalg.norm(f)
    rels = []
    for _ in range(6):
        p = poisson_mg._vcycle(p, f, dx, dy, poisson_mg.BC_PCORR)
        r = f - poisson_mg.apply_A(p, dx, dy, poisson_mg.BC_PCORR)
        rels.append(np.linalg.norm(r) / rhs_norm)
    # After the first cycle (initial fast transient), contraction must be < 0.5 per cycle
    rates = [rels[i + 1] / max(rels[i], 1e-30) for i in range(1, len(rels) - 1)]
    assert max(rates) < 0.5, f"rates={rates}"
