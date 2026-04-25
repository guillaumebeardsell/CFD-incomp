"""Micro-benchmark harness for solver-optimization work.

Times a steady cylinder-in-tunnel solve at the three FIDELITY sizes,
saves a JSON summary so baseline and optimized runs can be diffed.

Usage:
    python -m benchmarks.bench_opt --out baseline.json
    # (apply optimizations)
    python -m benchmarks.bench_opt --out optimized.json
    python -m benchmarks.bench_opt --diff baseline.json optimized.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.solver import Solver, SolverConfig, FIDELITY
from app.solver import poisson_mg
from app.solver.mesh import build_mesh, rasterize_disk


# --- problem setup (matches the UI scenario: 20x10 tunnel, D=1 cylinder) ---
DOMAIN = (20.0, 10.0)
CYL_CENTER = (5.0, 5.0)
D = 1.0
U_INF = 1.0
RE = 40.0


def _square_cell_grid(width: float, height: float, nx_d: int, ny_d: int):
    """Replicates app._parse_spec: pick the finer of w/nx, h/ny and match other."""
    dx = min(width / nx_d, height / ny_d)
    nx = max(16, int(round(width / dx)))
    ny = max(16, int(round(height / dx)))
    if nx % 2: nx += 1
    if ny % 2: ny += 1
    return nx, ny


def _run(nx: int, ny: int, max_iters: int, res_drop: float):
    """Run a single solve; return dict of timings & stats."""
    # Patch poisson_mg.solve to count cycles. Restore at the end.
    orig_solve = poisson_mg.solve
    cycles_log: list[int] = []
    def wrapped(rhs, dx, dy, tol=1e-4, max_cycles=30, n_smooth=2, p0=None,
                bc_sides=poisson_mg.BC_PCORR):
        p, n_c, rel = orig_solve(rhs, dx, dy, tol=tol, max_cycles=max_cycles,
                                 n_smooth=n_smooth, p0=p0, bc_sides=bc_sides)
        cycles_log.append(n_c)
        return p, n_c, rel
    poisson_mg.solve = wrapped

    try:
        mesh = build_mesh(*DOMAIN, nx, ny)
        chi = rasterize_disk(*CYL_CENTER, 0.5 * D, mesh)
        bcs = {
            "inlet":  {"type": "inlet_velocity", "speed": U_INF, "angle_deg": 0.0},
            "outlet": {"type": "outlet_pressure", "p": 0.0},
            "top":    {"type": "slip"},
            "bottom": {"type": "slip"},
        }
        cfg = SolverConfig(mode="steady", Re=RE, U_ref=U_INF, L_ref=D, eta=1e-3,
                           cfl_steady=0.5, alpha_p_steady=0.7,
                           max_iters=max_iters, res_drop=res_drop,
                           log_interval=10_000_000)  # silence
        solver = Solver(mesh, chi, bcs, cfg, log=lambda _s: None)
        t0 = time.perf_counter()
        r = solver.solve()
        elapsed = time.perf_counter() - t0

        # Extract a "wake velocity" sanity check: u at centerline, 1D downstream of cyl
        jc = ny // 2
        ic = int(round((CYL_CENTER[0] + 1.5 * D) / mesh["dx"]))
        u_wake = float(r.u[jc, ic])

        # Iter at which R_mom first drops below thresholds (for time-to-solution)
        def _first_below(thresh: float) -> int:
            for k, rm in enumerate(r.residuals):
                if rm < thresh:
                    return k
            return -1
        iter_below_1e_1 = _first_below(1e-1)
        iter_below_1e_2 = _first_below(1e-2)
        iter_below_1e_3 = _first_below(1e-3)

        return {
            "nx": nx, "ny": ny,
            "cells": nx * ny,
            "iters": int(r.iters),
            "elapsed_s": float(elapsed),
            "s_per_iter_ms": 1000.0 * elapsed / max(r.iters, 1),
            "converged": bool(r.converged),
            "final_R_mom": float(r.residuals[-1]) if r.residuals else float("nan"),
            "mg_total_cycles": int(sum(cycles_log)),
            "mg_mean_cycles": float(np.mean(cycles_log)) if cycles_log else 0.0,
            "iter_below_1e_1": iter_below_1e_1,
            "iter_below_1e_2": iter_below_1e_2,
            "iter_below_1e_3": iter_below_1e_3,
            "u_wake": u_wake,
        }
    finally:
        poisson_mg.solve = orig_solve


def run_all(max_iters: int, res_drop: float) -> dict:
    results = {}
    for tier, (nx_d, ny_d) in FIDELITY.items():
        nx, ny = _square_cell_grid(*DOMAIN, nx_d, ny_d)
        print(f"[{tier}] grid {nx}x{ny} = {nx*ny} cells ...")
        r = _run(nx, ny, max_iters=max_iters, res_drop=res_drop)
        print(f"  iters={r['iters']}  elapsed={r['elapsed_s']:.2f}s  "
              f"({r['s_per_iter_ms']:.2f} ms/iter)  "
              f"mg_mean={r['mg_mean_cycles']:.2f}  "
              f"R_mom={r['final_R_mom']:.2e}  u_wake={r['u_wake']:.3f}")
        results[tier] = r
    return results


def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


def diff_and_print(base_path: str, opt_path: str):
    base = json.loads(Path(base_path).read_text())
    opt = json.loads(Path(opt_path).read_text())
    print("\n== Per-iter cost (same iter budget) ==")
    print(f"{'tier':<8}{'grid':>12}{'base t':>10}{'opt t':>10}{'speedup':>10}"
          f"{'base ms/it':>12}{'opt ms/it':>12}{'base mg/it':>12}{'opt mg/it':>12}")
    print("-" * 98)
    for tier in FIDELITY:
        b = base[tier]; o = opt[tier]
        speedup = b["s_per_iter_ms"] / o["s_per_iter_ms"]
        grid = f"{b['nx']}x{b['ny']}"
        print(f"{tier:<8}{grid:>12}{b['elapsed_s']:>9.1f}s{o['elapsed_s']:>9.1f}s"
              f"{speedup:>9.2f}x{b['s_per_iter_ms']:>12.2f}{o['s_per_iter_ms']:>12.2f}"
              f"{b['mg_mean_cycles']:>12.2f}{o['mg_mean_cycles']:>12.2f}")

    print("\n== Convergence progress in same iter budget ==")
    print(f"{'tier':<8}{'base R_mom':>14}{'opt R_mom':>14}"
          f"{'base→1e-2':>14}{'opt→1e-2':>14}{'base→1e-3':>14}{'opt→1e-3':>14}")
    print("-" * 96)
    def _fmt_it(v):
        return f"{v}" if v >= 0 else "—"
    for tier in FIDELITY:
        b = base[tier]; o = opt[tier]
        print(f"{tier:<8}{b['final_R_mom']:>14.2e}{o['final_R_mom']:>14.2e}"
              f"{_fmt_it(b['iter_below_1e_2']):>14}{_fmt_it(o['iter_below_1e_2']):>14}"
              f"{_fmt_it(b['iter_below_1e_3']):>14}{_fmt_it(o['iter_below_1e_3']):>14}")
        du = abs(b["u_wake"] - o["u_wake"])
        if du > 0.05:
            print(f"  WARN: u_wake drifted by {du:.3f} ({b['u_wake']:.3f} -> {o['u_wake']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, help="write JSON results to this path")
    ap.add_argument("--diff", nargs=2, metavar=("BASE", "OPT"),
                    help="print speedup table from two result files")
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--res-drop", type=float, default=1e-4)
    args = ap.parse_args()

    if args.diff:
        diff_and_print(*args.diff)
    else:
        results = run_all(max_iters=args.max_iters, res_drop=args.res_drop)
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2))
            print(f"\nwrote {args.out}")
