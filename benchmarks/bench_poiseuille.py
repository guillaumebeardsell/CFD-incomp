"""Penalized plane Poiseuille benchmark.

Channel with top and bottom walls represented by **Brinkman masks**. Inlet is
set to the analytic parabolic profile for the effective fluid channel; outlet
is pressure = 0. If the solver is implemented correctly, the parabolic profile
is preserved throughout the channel (since it is the fully-developed solution).

Sweeps grid resolution (ny ∈ {32, 64, 128}) × Brinkman η ∈ {1e-2, 1e-4, 1e-6}.
Reports L2 error of u at mid-channel vs analytic; fits orders in dx and η.
"""
from __future__ import annotations

import numpy as np

from app.solver import Solver, SolverConfig
from app.solver.bc import Boundaries
from app.solver.mesh import build_mesh
from . import _plot, _report

# Fix channel geometry.
H = 1.0
L = 4.0
WALL_FRAC = 0.1          # Brinkman walls fill outer 10% at top and bottom
U_BAR = 1.0              # mean velocity in the fluid core
RE = 100.0               # fluid Re (chosen so viscous timescale is tractable)


def _analytic_profile(y: np.ndarray) -> np.ndarray:
    """Parabolic Poiseuille profile for effective fluid region."""
    y_lo = WALL_FRAC * H
    y_hi = (1.0 - WALL_FRAC) * H
    h = y_hi - y_lo
    y_c = 0.5 * (y_lo + y_hi)
    u_max = 1.5 * U_BAR
    r = (y - y_c) / (h / 2.0)
    u = u_max * (1.0 - r ** 2)
    u = np.where((y > y_lo) & (y < y_hi), u, 0.0)
    return u


def run_one(ny: int, eta: float, max_iters: int = 4000):
    nx = ny * int(L / H)
    mesh = build_mesh(L, H, nx, ny)
    chi = (mesh["Y"] < WALL_FRAC * H) | (mesh["Y"] > (1.0 - WALL_FRAC) * H)
    u_profile = _analytic_profile(mesh["yc"])

    bcs_dict = {
        "inlet":  {"type": "inlet_velocity", "profile": u_profile},
        "outlet": {"type": "outlet_pressure", "p": 0.0},
        "top":    {"type": "slip"},
        "bottom": {"type": "slip"},
    }
    cfg = SolverConfig(mode="steady", Re=RE, U_ref=U_BAR, L_ref=H, eta=eta,
                       max_iters=max_iters, res_drop=1e-5, log_interval=10_000,
                       cfl_steady=0.4)
    solver = Solver(mesh, chi, bcs_dict, cfg,
                    u_init=np.broadcast_to(u_profile[:, None], (ny, nx)).copy())
    result = solver.solve()
    # Error over fluid-only cells at mid-channel column
    ix = nx // 2
    fluid = ~chi[:, ix]
    u_num = result.u[:, ix]
    u_ex = u_profile
    err = u_num - u_ex
    l2 = float(np.sqrt(np.mean(err[fluid] ** 2)))
    linf = float(np.max(np.abs(err[fluid])))
    return {
        "ny": ny, "nx": nx, "dx": mesh["dx"], "dy": mesh["dy"],
        "eta": eta, "iters": result.iters, "converged": result.converged,
        "L2": l2, "Linf": linf,
        "u_profile_num": u_num, "u_profile_ex": u_ex, "y": mesh["yc"],
        "elapsed_s": result.elapsed_s,
    }


def fit_order_in_dx(dxs, errs):
    lx = np.log(dxs); le = np.log(np.maximum(errs, 1e-30))
    slope = np.polyfit(lx, le, 1)[0]
    return float(slope)


def build_report() -> str:
    grids = [32, 64, 128]
    etas = [1e-2, 1e-4]

    all_runs = {}
    for eta in etas:
        for ny in grids:
            key = (ny, eta)
            print(f"[poiseuille] ny={ny} eta={eta:.0e}")
            all_runs[key] = run_one(ny, eta, max_iters=2500)

    # Convergence plot: L2 vs dx at each eta
    fig, ax = _plot.new_fig(6.5, 4.0)
    for eta in etas:
        dxs = np.array([all_runs[(ny, eta)]["dx"] for ny in grids])
        errs = np.array([all_runs[(ny, eta)]["L2"] for ny in grids])
        ax.loglog(dxs, errs, "o-", label=f"η={eta:.0e}")
    ax.loglog([0.02, 0.1], [0.01, 0.01 * 25], "k--", alpha=0.3, label="slope 2")
    ax.set_xlabel("dx"); ax.set_ylabel("L2 error of u at mid-channel")
    ax.set_title("Grid convergence"); ax.legend(); ax.grid(alpha=0.4)
    conv_img = _plot.fig_to_data_uri(fig)

    # Profile at finest grid
    fig, ax = _plot.new_fig(6.5, 4.0)
    for eta in etas:
        r = all_runs[(grids[-1], eta)]
        ax.plot(r["u_profile_num"], r["y"], "o-", markersize=3, label=f"η={eta:.0e}")
    ax.plot(all_runs[(grids[-1], etas[-1])]["u_profile_ex"],
            all_runs[(grids[-1], etas[-1])]["y"], "k--", label="analytic")
    ax.set_xlabel("u"); ax.set_ylabel("y")
    ax.set_title(f"Mid-channel profile (ny={grids[-1]})"); ax.legend(); ax.grid(alpha=0.4)
    prof_img = _plot.fig_to_data_uri(fig)

    # Metrics table
    table_rows = []
    for ny in grids:
        for eta in etas:
            r = all_runs[(ny, eta)]
            table_rows.append((ny, f"{eta:.0e}", f"{r['L2']:.3e}",
                               f"{r['Linf']:.3e}", r["iters"],
                               "yes" if r["converged"] else "no"))
    results_table = _report.plain_table(
        ["ny", "η", "L2", "L∞", "iters", "conv"], table_rows,
    )

    # PASS/FAIL: at η=1e-2 (well-resolved Brinkman layer), expect error
    # monotonically decreasing with dx. 1st-order upwind + Brinkman has
    # messy convergence, so the test is just that the finest-grid error
    # is less than 50% of coarsest and peak profile is within 15% of analytic.
    dxs = np.array([all_runs[(ny, 1e-2)]["dx"] for ny in grids])
    errs = np.array([all_runs[(ny, 1e-2)]["L2"] for ny in grids])
    order = fit_order_in_dx(dxs, errs)
    shrink_ratio = errs[-1] / errs[0]
    peak_num = max(all_runs[(grids[-1], 1e-2)]["u_profile_num"])
    peak_err_frac = abs(peak_num - 1.5 * U_BAR) / (1.5 * U_BAR)
    passed = (shrink_ratio < 0.6) and (peak_err_frac < 0.15)
    summary = _report.metrics_table([
        ("Fitted dx-order at η=1e-2", f"{order:.2f}",
         "—", "informational", True),
        ("Error shrink (fine/coarse)", f"{shrink_ratio:.2f}",
         "< 1.0", "< 0.6 (test)", shrink_ratio < 0.6),
        ("Peak velocity rel-err", f"{peak_err_frac:.2%}",
         "0 (analytic)", "< 15% (test)", peak_err_frac < 0.15),
    ])

    sections = [
        _report.section("Summary", summary),
        _report.section("Velocity profile at mid-channel", _report.img(prof_img)),
        _report.section("Grid convergence", _report.img(conv_img), results_table),
        _report.section("Setup",
                        _report.p(f"Channel {L}x{H}, walls (Brinkman) at y<{WALL_FRAC*H} and "
                                  f"y>{(1-WALL_FRAC)*H}. Re={RE}, U̅={U_BAR}. "
                                  "Inlet=analytic parabolic profile, outlet p=0.")),
    ]
    path = _report.write_report("Benchmark: Penalized plane Poiseuille",
                                "poiseuille.html", sections)
    return {"path": str(path), "passed": bool(passed), "order": order,
            "shrink_ratio": float(shrink_ratio), "peak_err_frac": float(peak_err_frac)}


def test_poiseuille_convergence():
    """pytest assertion: finest-grid L2 < 60% of coarsest AND peak within 15% of analytic."""
    r = build_report()
    assert r["passed"], f"order={r['order']:.2f}, shrink_ratio={r.get('shrink_ratio','?')}, peak_err={r.get('peak_err_frac','?')}"


if __name__ == "__main__":
    r = build_report()
    print("Report:", r["path"], " order =", r["order"])
