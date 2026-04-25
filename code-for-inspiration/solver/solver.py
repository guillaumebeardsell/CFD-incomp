"""Steady-state 2D compressible Navier-Stokes solver on a Cartesian grid with
immersed-boundary (staircase) treatment of an arbitrary obstacle.

Numerics:
    - Finite volume on a uniform Cartesian grid
    - MUSCL reconstruction with minmod limiter (2nd-order spatial)
    - Roe approximate Riemann solver with Harten entropy fix (inviscid)
    - Optional central-difference viscous fluxes (Sutherland viscosity)
    - RK3 SSP time integration with local timestepping
    - Pseudo-time marching until density residual drops RES_DROP_DEFAULT
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from . import bc as bc_mod
from . import config, mesh as mesh_mod, state as state_mod
from . import immersed as ib_mod
from .flux_inviscid import roe_flux_x, roe_flux_y
from .reconstruct import muscl_faces
from .smoothing import smooth_residual

try:
    from .flux_viscous import viscous_flux_x, viscous_flux_y
    HAS_VISCOUS = True
except Exception:
    HAS_VISCOUS = False


@dataclass
class SolveResult:
    W: np.ndarray          # primitive fields, shape (4, ny, nx): rho, u, v, p
    T: np.ndarray          # temperature (ny, nx)
    mach: np.ndarray       # Mach number (ny, nx)
    res_field: np.ndarray  # log10(|d rho/dt|) per cell (ny, nx) at the last step
    mask: np.ndarray       # bool solid mask (ny, nx)
    mesh: dict
    residuals: list
    iters: int
    converged: bool
    # Transient-mode only (None in steady mode): stacked arrays of shape
    # (n_frames, ny, nx) sampled over the 2nd flow-through; plus the sample
    # physical times.
    frames_mach: np.ndarray = None
    frames_u: np.ndarray = None
    frames_v: np.ndarray = None
    frame_times: np.ndarray = None


class Solver:
    def __init__(
        self,
        mesh,
        solid_mask,
        bcs,
        gas=None,
        *,
        viscous=False,
        max_iters=config.MAX_ITERS_DEFAULT,
        res_drop=config.RES_DROP_DEFAULT,
        cfl=config.CFL,
        log=None,
        progress=None,
        polygon_xy=None,
        scheme="explicit",
        mode="steady",
    ):
        if scheme not in ("explicit", "irs"):
            raise ValueError(f"unknown scheme: {scheme!r}")
        if mode not in ("steady", "transient"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "transient" and scheme == "irs":
            raise ValueError("IRS is time-inaccurate; not usable in transient mode")
        self.mode = mode
        self.mesh = mesh
        self.solid = solid_mask.astype(bool)
        self.bcs = bcs
        self.gas = gas or {"gamma": config.GAMMA, "R": config.R_GAS}
        self.gamma = self.gas["gamma"]
        self.R = self.gas["R"]
        self.viscous = viscous and HAS_VISCOUS
        self.max_iters = max_iters
        self.res_drop = res_drop
        self.cfl = cfl
        self.scheme = scheme
        self.eps_irs = config.EPS_IRS if scheme == "irs" else 0.0
        self.log = log
        self.progress = progress

        nx, ny = mesh["nx"], mesh["ny"]
        self.nx, self.ny = nx, ny
        self.dx, self.dy = mesh["dx"], mesh["dy"]

        (
            self.xf_type,
            self.xf_fluid_side,
            self.yf_type,
            self.yf_fluid_side,
        ) = mesh_mod.face_masks(self.solid)

        # Ghost-cell IB: when a polygon is supplied, fluid/solid faces are
        # handled by reflecting fluid state into the body across the true
        # boundary rather than by the staircase wall-flux patch.
        self.obstacle_bc = bcs.get("obstacle", {"type": "slip_wall"})
        self.ghost_info = ib_mod.build_ghost_info(polygon_xy, mesh, self.solid) if polygon_xy else None
        self.use_ib = self.ghost_info is not None
        # The ghost-cell reflection can double the effective velocity jump at
        # fluid/ghost faces (especially for no-slip), which makes a fluid cell's
        # local dt from its own state optimistic. Tighten CFL accordingly.
        if self.use_ib and self.scheme == "explicit":
            self.cfl = min(self.cfl, 0.3)
        if self.scheme == "irs":
            self.cfl = config.CFL_IRS_IB if self.use_ib else config.CFL_IRS

        # Padded conservative state (4, ny+4, nx+4).
        self.U = self._initial_condition()

    # ------------------------------------------------------------------ init
    def _freestream_primitive(self):
        ib = self.bcs["inlet"]
        if ib["type"] in ("inlet_subsonic", "inlet_supersonic"):
            p = ib["p"]
            T = ib["T"]
            rho = p / (self.R * T)
            c = np.sqrt(self.gamma * p / rho)
            u = ib["mach"] * c
            return np.array([rho, u, 0.0, p])
        # Fallback: standard sea-level air at Mach 0.3
        T, p = 300.0, 101325.0
        rho = p / (self.R * T)
        c = np.sqrt(self.gamma * p / rho)
        return np.array([rho, 0.3 * c, 0.0, p])

    def _initial_condition(self):
        W_inf = self._freestream_primitive()
        nx, ny = self.nx, self.ny
        W = np.zeros((4, ny + 4, nx + 4))
        for k in range(4):
            W[k] = W_inf[k]
        U = state_mod.primitive_to_conservative(W, self.gamma)
        return U

    # ------------------------------------------------------------------ core
    def _compute_rhs(self, U):
        """Return dU/dt from flux divergence for the fluid interior (4, ny, nx)."""
        W = state_mod.conservative_to_primitive(U, self.gamma)
        bc_mod.apply_bcs(W, self.bcs, gamma=self.gamma, R=self.R)
        if self.use_ib:
            ib_mod.fill_ghosts(W, self.ghost_info, self.obstacle_bc, self.R)

        # --- X-faces: MUSCL reconstruction, Roe flux, IB correction -----
        WL_x, WR_x = muscl_faces(W, axis=2)  # (4, ny+4, nx+1)
        F_x = roe_flux_x(WL_x, WR_x, self.gamma)  # same shape

        # Slice to fluid y-range -> (4, ny, nx+1)
        F_x = F_x[:, 2:2 + self.ny, :]
        WL_x_f = WL_x[:, 2:2 + self.ny, :]
        WR_x_f = WR_x[:, 2:2 + self.ny, :]

        wall = self.xf_type == 1
        solid_solid = self.xf_type == 2
        if not self.use_ib:
            # Staircase wall flux: zero mass/energy, pressure-only momentum.
            p_wall = np.where(self.xf_fluid_side == 0, WL_x_f[3], WR_x_f[3])
            F_x[0][wall] = 0.0
            F_x[1][wall] = p_wall[wall]
            F_x[2][wall] = 0.0
            F_x[3][wall] = 0.0
        F_x[:, solid_solid] = 0.0

        # --- Y-faces ----------------------------------------------------
        WL_y, WR_y = muscl_faces(W, axis=1)  # (4, ny+1, nx+4)
        F_y = roe_flux_y(WL_y, WR_y, self.gamma)
        F_y = F_y[:, :, 2:2 + self.nx]
        WL_y_f = WL_y[:, :, 2:2 + self.nx]
        WR_y_f = WR_y[:, :, 2:2 + self.nx]
        wall_y = self.yf_type == 1
        solid_solid_y = self.yf_type == 2
        if not self.use_ib:
            p_wall_y = np.where(self.yf_fluid_side == 0, WL_y_f[3], WR_y_f[3])
            F_y[0][wall_y] = 0.0
            F_y[1][wall_y] = 0.0
            F_y[2][wall_y] = p_wall_y[wall_y]
            F_y[3][wall_y] = 0.0
        F_y[:, solid_solid_y] = 0.0

        # --- Viscous fluxes ---------------------------------------------
        if self.viscous:
            Wv = W
            if not self.use_ib:
                # Staircase no-slip: force zero velocity inside solid to create
                # a wall-normal gradient at fluid/solid faces.
                Wv = W.copy()
                Wv[1, 2:-2, 2:-2][self.solid] = 0.0
                Wv[2, 2:-2, 2:-2][self.solid] = 0.0
            Gx = viscous_flux_x(Wv, self.dx, self.dy, self.gamma, self.R)
            Gy = viscous_flux_y(Wv, self.dx, self.dy, self.gamma, self.R)
            Gx[:, solid_solid] = 0.0
            Gy[:, solid_solid_y] = 0.0
            F_x -= Gx
            F_y -= Gy

        # --- Divergence -------------------------------------------------
        dFx = (F_x[:, :, 1:] - F_x[:, :, :-1]) / self.dx          # (4, ny, nx)
        dFy = (F_y[:, 1:, :] - F_y[:, :-1, :]) / self.dy          # (4, ny, nx)
        rhs = -(dFx + dFy)

        # Zero out residuals in solid cells
        rhs[:, self.solid] = 0.0
        return rhs, W

    def _local_dt(self, W):
        """(ny, nx) local timestep from CFL condition."""
        rho = W[0, 2:-2, 2:-2]
        u = W[1, 2:-2, 2:-2]
        v = W[2, 2:-2, 2:-2]
        p = W[3, 2:-2, 2:-2]
        c = np.sqrt(self.gamma * p / rho)
        denom = (np.abs(u) + c) / self.dx + (np.abs(v) + c) / self.dy
        if self.use_ib:
            # At fluid/ghost faces the ghost carries a reflected velocity, so
            # the face's effective wave speed can exceed the fluid cell's own.
            # Take a 3x3 max of denom so each fluid cell's dt is safe for all
            # incident faces.
            pad = np.pad(denom, 1, mode="edge")
            denom = np.maximum.reduce([
                pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
                pad[1:-1, 0:-2], pad[1:-1, 1:-1], pad[1:-1, 2:],
                pad[2:,   0:-2], pad[2:,   1:-1], pad[2:,   2:],
            ])
        dt = self.cfl / np.maximum(denom, 1e-30)
        dt[self.solid] = 0.0
        return dt

    def _global_dt(self, W):
        """Scalar dt (broadcast as (ny, nx)) from the most restrictive cell."""
        local = self._local_dt(W)
        # Exclude solid cells (dt=0 there) from the min.
        fluid = local[~self.solid]
        dt_s = float(np.min(fluid)) if fluid.size else 0.0
        out = np.full_like(local, dt_s)
        out[self.solid] = 0.0
        return out

    def _rk3_step(self, U):
        rhs1, W = self._compute_rhs(U)
        dt = self._global_dt(W) if self.mode == "transient" else self._local_dt(W)
        rhs1_apply = self._smooth(rhs1)
        # Broadcast dt (ny,nx) to (1, ny+4, nx+4) slice layout by applying only to interior.
        def apply(U_in, rhs, factor):
            out = U_in.copy()
            out[:, 2:-2, 2:-2] += factor * dt * rhs
            return out

        U1 = apply(U, rhs1_apply, 1.0)
        rhs2, _ = self._compute_rhs(U1)
        rhs2 = self._smooth(rhs2)
        U2 = U.copy()
        U2[:, 2:-2, 2:-2] = 0.75 * U[:, 2:-2, 2:-2] + 0.25 * (U1[:, 2:-2, 2:-2] + dt * rhs2)
        rhs3, _ = self._compute_rhs(U2)
        rhs3 = self._smooth(rhs3)
        U3 = U.copy()
        U3[:, 2:-2, 2:-2] = (1.0 / 3.0) * U[:, 2:-2, 2:-2] + (2.0 / 3.0) * (
            U2[:, 2:-2, 2:-2] + dt * rhs3
        )
        # dt_scalar is meaningful only in transient mode (uniform dt everywhere).
        dt_scalar = float(dt.max()) if self.mode == "transient" else 0.0
        return U3, rhs1, dt_scalar

    def _smooth(self, rhs):
        if self.scheme != "irs":
            return rhs
        r = smooth_residual(rhs, self.eps_irs)
        r[:, self.solid] = 0.0
        return r

    def solve(self) -> SolveResult:
        if self.mode == "transient":
            return self._solve_transient()
        return self._solve_steady()

    def _solve_steady(self) -> SolveResult:
        residuals = []
        start = time.time()
        initial_res = None
        converged = False
        it = 0
        for it in range(1, self.max_iters + 1):
            self.U, rhs, _ = self._rk3_step(self.U)
            if not np.all(np.isfinite(self.U)):
                raise RuntimeError(f"Non-finite state at iter {it}")
            res = float(np.sqrt(np.mean(rhs[0] ** 2)))
            residuals.append(res)
            if initial_res is None and res > 0.0:
                initial_res = res
            if initial_res and res / initial_res < self.res_drop:
                converged = True
                break
            if it % config.LOG_INTERVAL == 0:
                if self.log:
                    self.log(f"iter {it:6d}  res(rho)={res:.3e}")
                if self.progress:
                    self.progress(it, res)
        elapsed = time.time() - start
        if self.log:
            self.log(f"done: iter {it} elapsed {elapsed:.1f}s converged={converged}")

        W = state_mod.conservative_to_primitive(self.U, self.gamma)[:, 2:-2, 2:-2]
        T = W[3] / (W[0] * self.R)
        c = np.sqrt(self.gamma * W[3] / W[0])
        mach = np.sqrt(W[1] ** 2 + W[2] ** 2) / c

        # Per-cell log10(|d rho/dt|) at the final state for residual heatmap.
        final_rhs, _ = self._compute_rhs(self.U)
        res_field = np.log10(np.abs(final_rhs[0]) + 1e-30)

        return SolveResult(
            W=W, T=T, mach=mach, res_field=res_field,
            mask=self.solid, mesh=self.mesh,
            residuals=residuals, iters=it, converged=converged,
        )

    def _solve_transient(self) -> SolveResult:
        """March in physical time with a global dt. Integrate for 2 flow-through
        times; record N_FRAMES snapshots over the 2nd pass for animation."""
        W_inf = self._freestream_primitive()
        U_inf = float(W_inf[1])  # inlet x-velocity
        width = self.nx * self.dx
        T_pass = width / max(U_inf, 1e-12)
        T_end = 2.0 * T_pass
        N = config.TRANSIENT_FRAMES

        frames_mach = np.zeros((N, self.ny, self.nx), dtype=np.float32)
        frames_u    = np.zeros((N, self.ny, self.nx), dtype=np.float32)
        frames_v    = np.zeros((N, self.ny, self.nx), dtype=np.float32)
        frame_times = np.zeros(N, dtype=np.float32)
        next_sample = 0
        # Sample targets uniformly in [T_pass, 2*T_pass).
        sample_targets = T_pass + (np.arange(N) / N) * T_pass

        residuals = []
        t = 0.0
        start = time.time()
        it = 0
        for it in range(1, config.MAX_TRANSIENT_ITERS + 1):
            self.U, rhs, dt = self._rk3_step(self.U)
            if not np.all(np.isfinite(self.U)):
                raise RuntimeError(f"Non-finite state at iter {it}")
            t += dt
            res = float(np.sqrt(np.mean(rhs[0] ** 2)))
            residuals.append(res)

            # Sample frames whose target time has been reached.
            while next_sample < N and t >= sample_targets[next_sample]:
                W_now = state_mod.conservative_to_primitive(self.U, self.gamma)[:, 2:-2, 2:-2]
                rho, u, v, p = W_now[0], W_now[1], W_now[2], W_now[3]
                c = np.sqrt(self.gamma * p / rho)
                frames_mach[next_sample] = (np.sqrt(u * u + v * v) / c).astype(np.float32)
                frames_u[next_sample] = u.astype(np.float32)
                frames_v[next_sample] = v.astype(np.float32)
                frame_times[next_sample] = t
                next_sample += 1

            if it % config.LOG_INTERVAL == 0:
                if self.log:
                    self.log(f"iter {it:6d}  t={t:.4f}/{T_end:.4f}s  dt={dt:.2e}  res={res:.3e}")
                if self.progress:
                    # Report progress as iter-equivalent for the UI.
                    self.progress(it, res)

            if t >= T_end and next_sample >= N:
                break
        else:
            raise RuntimeError(
                f"transient solve did not finish within MAX_TRANSIENT_ITERS={config.MAX_TRANSIENT_ITERS}"
            )

        elapsed = time.time() - start
        if self.log:
            self.log(f"transient done: {it} iters, t={t:.3f}s, elapsed {elapsed:.1f}s")

        # Final steady-like fields for the result (shown if animation idle).
        W = state_mod.conservative_to_primitive(self.U, self.gamma)[:, 2:-2, 2:-2]
        Tf = W[3] / (W[0] * self.R)
        c = np.sqrt(self.gamma * W[3] / W[0])
        mach = np.sqrt(W[1] ** 2 + W[2] ** 2) / c
        final_rhs, _ = self._compute_rhs(self.U)
        res_field = np.log10(np.abs(final_rhs[0]) + 1e-30)

        return SolveResult(
            W=W, T=Tf, mach=mach, res_field=res_field,
            mask=self.solid, mesh=self.mesh,
            residuals=residuals, iters=it, converged=True,
            frames_mach=frames_mach, frames_u=frames_u, frames_v=frames_v,
            frame_times=frame_times,
        )


# ---------------------------------------------------------------------- CLI

def _case_wedge():
    """Supersonic wedge test: M=2 flow over a 10-degree wedge (for validation)."""
    mesh = mesh_mod.build_mesh(width=2.0, height=1.0, nx=160, ny=80)
    # Wedge polygon: tip at (0.5, 0), angle 10 deg going right.
    import math
    theta = math.radians(10.0)
    x0, y0 = 0.5, 0.0
    x1 = 2.1
    y1 = (x1 - x0) * math.tan(theta)
    poly = [(x0, y0 - 0.01), (x1, y0 - 0.01), (x1, y1)]
    solid = mesh_mod.rasterize_polygon(poly, mesh)
    bcs = {
        "inlet":   {"type": "inlet_supersonic", "mach": 2.0, "p": 101325.0, "T": 300.0},
        "outlet":  {"type": "outlet_supersonic"},
        "top":     {"type": "slip_wall"},
        "bottom":  {"type": "slip_wall"},
    }
    return mesh, solid, bcs, poly


def _case_cylinder():
    """Subsonic cylinder test: M=0.3 flow past a circular cylinder."""
    mesh = mesh_mod.build_mesh(width=3.0, height=1.5, nx=240, ny=120)
    cx, cy, r = 1.0, 0.75, 0.15
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    poly = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in theta]
    solid = mesh_mod.rasterize_polygon(poly, mesh)
    bcs = {
        "inlet":   {"type": "inlet_subsonic", "mach": 0.3, "p": 101325.0, "T": 300.0},
        "outlet":  {"type": "outlet_subsonic", "p": 101325.0},
        "top":     {"type": "slip_wall"},
        "bottom":  {"type": "slip_wall"},
    }
    return mesh, solid, bcs, poly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["wedge", "cylinder"], default="cylinder")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--res-drop", type=float, default=1e-4)
    parser.add_argument("--viscous", action="store_true")
    args = parser.parse_args()

    mesh, solid, bcs, poly = _case_wedge() if args.case == "wedge" else _case_cylinder()
    solver = Solver(
        mesh, solid, bcs,
        viscous=args.viscous,
        max_iters=args.max_iters,
        res_drop=args.res_drop,
        log=print,
        polygon_xy=poly,
    )
    result = solver.solve()
    print(f"final residual = {result.residuals[-1]:.3e} after {result.iters} iters")
    print(f"max Mach = {result.mach.max():.3f},  min pressure = {result.W[3].min():.1f} Pa")


if __name__ == "__main__":
    main()
