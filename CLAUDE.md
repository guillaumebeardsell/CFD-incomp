# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install deps: `pip install -r requirements.txt` (numpy, scipy, shapely, flask, pytest, cantera, pillow, openpyxl, awscli, boto3).

Run the web app (Flask + static UI): `python -m app.app` (or `python app/app.py`). Serves on `PORT` (default 5150); UI at `/`, REST under `/api/solve`.

Run tests: `pytest` from repo root. Single test: `pytest tests/test_solver_smoke.py::test_cylinder_runs_and_stabilizes -q`. Tests import `app.solver`, so run from repo root (no installed package).

Run verification benchmarks: `python -m benchmarks.run_all [coarse|medium]`. Writes HTML reports + an index to `benchmarks/reports/`. Individual: `python -m benchmarks.bench_poiseuille`, `python -m benchmarks.bench_cylinder_re40`.

## Architecture

Incompressible 2D Navier–Stokes solver on a collocated cell-centered Cartesian grid, served behind a small Flask API with a browser UI for drawing obstacles.

**Solver pipeline (`app/solver/`)** — one time step is predictor → projection:
- `predictor.py`: first-order upwind advection + central diffusion + Brinkman penalization (solid cells driven to zero velocity via `eta`); returns `u*, v*` and the diagonal `a_P` of the implicit momentum operator.
- `projection.py`: Rhie–Chow face interpolation (`operators.rhie_chow_faces`) to kill checkerboard pressure on the collocated grid, Poisson solve for `p'`, then velocity correction `u = u* − dt·∇p'/a_P` and pressure update `p += alpha_p·p'`.
- `poisson_mg.py`: geometric multigrid V-cycle. BC sides are tagged `DIRICHLET` (ghost = −interior) or `NEUMANN` (ghost = interior) per-edge; pressure-correction uses Dirichlet at the outlet, Neumann elsewhere. The app enforces even `nx, ny` and (in `_parse_spec`) square cells so coarsening stays well-conditioned.
- `solver.py` (`Solver`): orchestrates two modes. `steady` is pseudo-transient with pressure under-relaxation `alpha_p_steady` and a residual-flattening convergence check (`residuals.flattened` over `conv_window`). `transient` first runs a steady warm-start (halved iters, loosened `res_drop`), then integrates physical time with `alpha_p=1`, skipping `t_buffer` seconds before recording frames every `frame_dt` (frames capped at `frame_dt/4` for CFL).
- `bc.py` (`Boundaries`): owns ghost-cell padding for u, v, p. Inlet is velocity (optional per-row `profile` overrides `speed`/`angle_deg`); outlet is pressure Dirichlet; top/bottom are `slip`/`symmetry`/`no_slip`. Obstacles are **not** BCs — they are Brinkman-penalized via the `chi` mask.
- `mesh.py`: `build_mesh` + `rasterize_polygon` (uses `shapely.vectorized.contains`; applies `buffer(0)` to repair hand-drawn polygons that are technically invalid).
- `state.py`, `diagnostics.py`, `residuals.py`, `operators.py`: field container, vorticity, residual metrics, finite-difference stencils.

**Flask backend (`app/app.py`)**: `/api/solve` POST kicks off a background thread and returns a `job_id`; `/api/solve/status` polls progress; `/api/solve/result` returns the payload once `status == "done"`. Field arrays are base64-encoded float32 (uint8 for the obstacle mask) for transport. At most 3 jobs are retained in memory. `FIDELITY` maps `"coarse"/"medium"/"fine"` → target grid sizes that then get squared and evened.

**Frontend (`app/static/`)**: plain HTML + vanilla JS (`draw.js` polygon capture, `bc.js` BC form, `solve.js` API driver, `render.js` field visualization, `app.js` entry). No build step.

**Benchmarks (`benchmarks/`)**: verification cases (plane Poiseuille convergence order, cylinder Re=40 vs Gautier reference) that import the solver directly and emit HTML via `_report.py` + `_plot.py`.

## Notes

- When running `python app/app.py` directly (not as a module), `app.py` inserts the repo root onto `sys.path` and falls back to `from solver import ...` — keep that dual-import pattern intact when touching imports.
- `code-for-inspiration/` and `literature/` are reference material, not part of the package.
