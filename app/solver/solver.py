"""Solver orchestrator: steady (pseudo-transient) and transient modes.

`Solver(mesh, chi, bcs, cfg).solve()` returns a SolveResult with field snapshots,
residual history, and (in transient mode) frame arrays suitable for the Flask
API payload.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import time

import numpy as np

from .bc import Boundaries
from .config import SolverConfig
from .diagnostics import vorticity
from .predictor import predictor
from .projection import project
from .residuals import momentum_residual, flattened
from .state import FieldState
from ._xp import xp, asarray, asnumpy


@dataclass
class SolveResult:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    vorticity: np.ndarray
    res_mom_field: np.ndarray
    res_div_field: np.ndarray
    mask: np.ndarray
    mesh: dict
    residuals: list
    cont_residuals: list
    iters: int
    converged: bool
    cancelled: bool
    elapsed_s: float
    mode: str
    # Transient-only (empty lists in steady)
    frames_u: list = field(default_factory=list)
    frames_v: list = field(default_factory=list)
    frames_p: list = field(default_factory=list)
    frame_times: list = field(default_factory=list)


class Solver:
    def __init__(self, mesh: dict, chi: np.ndarray, bcs: dict, cfg: SolverConfig,
                 log: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[dict], None]] = None,
                 residual_log: Optional[Callable[[int, float, float], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 u_init: Optional[np.ndarray] = None,
                 v_init: Optional[np.ndarray] = None,
                 p_init: Optional[np.ndarray] = None,
                 force_x: float = 0.0,
                 force_y: float = 0.0):
        self.mesh = mesh
        self.chi = asarray(np.asarray(chi, dtype=bool))
        self.boundaries = Boundaries(bcs)
        self.cfg = cfg
        self.log = log or (lambda _s: None)
        self.progress = progress
        self.residual_log = residual_log
        self.should_stop = should_stop or (lambda: False)
        self.cancelled = False
        self.force_x = float(force_x)
        self.force_y = float(force_y)
        self.phase: str = "steady"
        self._phase_max_iters: int = cfg.max_iters
        self._phase_res_drop: float = cfg.res_drop

        self.nu = cfg.U_ref * cfg.L_ref / cfg.Re
        self.dx = mesh["dx"]; self.dy = mesh["dy"]

        ny, nx = mesh["ny"], mesh["nx"]
        if u_init is None:
            ui = xp.asarray(self.boundaries.U_in)
            u0 = xp.broadcast_to(ui[:, None] if ui.ndim == 1 else ui, (ny, nx)).astype(xp.float64).copy()
            u0[self.chi] = 0.0
        else:
            u0 = asarray(u_init).copy()
        if v_init is None:
            vi = xp.asarray(self.boundaries.V_in)
            v0 = xp.broadcast_to(vi[:, None] if vi.ndim == 1 else vi, (ny, nx)).astype(xp.float64).copy()
            v0[self.chi] = 0.0
        else:
            v0 = asarray(v_init).copy()
        if p_init is None:
            p0 = xp.zeros((mesh["ny"], mesh["nx"]), dtype=xp.float64)
        else:
            p0 = asarray(p_init).copy()

        self.state = FieldState(u=u0, v=v0, p=p0, chi=self.chi)
        self._p_prime_prev: Optional[np.ndarray] = None

    # ---- Δt policies -------------------------------------------------------
    def _dt_steady(self, u, v) -> float:
        # Point-implicit diffusion in predictor: no explicit viscous dt cap.
        umax = float(xp.max(xp.sqrt(u * u + v * v)))
        conv = self.cfg.cfl_steady * min(self.dx, self.dy) / max(umax, 1e-6)
        return float(conv)

    def _dt_transient(self, u, v) -> float:
        umax = float(xp.max(xp.sqrt(u * u + v * v)))
        conv = self.cfg.cfl_transient * min(self.dx, self.dy) / max(umax, 1e-6)
        cap = self.cfg.frame_dt / 4.0
        return float(min(conv, cap))

    # ---- main entry --------------------------------------------------------
    def solve(self) -> SolveResult:
        t0 = time.time()
        converged = False
        if self.cfg.mode == "steady":
            self.phase = "steady"
            self._phase_max_iters = self.cfg.max_iters
            self._phase_res_drop = self.cfg.res_drop
            converged = self._loop_steady(
                max_iters=self.cfg.max_iters,
                res_drop=self.cfg.res_drop,
                alpha_p=self.cfg.alpha_p_steady,
            )
        elif self.cfg.mode == "transient":
            self.log("warm-start: steady phase")
            self.phase = "warm-start"
            self._phase_max_iters = max(1000, self.cfg.max_iters // 2)
            self._phase_res_drop = max(self.cfg.res_drop, 1e-3)
            self._loop_steady(
                max_iters=self._phase_max_iters,
                res_drop=self._phase_res_drop,
                alpha_p=self.cfg.alpha_p_steady,
            )
            if not self.cancelled:
                self.log("switching to physical-time integration")
                self.state.t = 0.0
                self.state.residuals = []
                self.state.cont_residuals = []
                self.phase = "transient"
                self._loop_transient()
            converged = not self.cancelled
        else:
            raise ValueError(f"unknown mode: {self.cfg.mode}")

        elapsed = time.time() - t0
        res_mom = (self.state.res_mom_field
                   if self.state.res_mom_field is not None
                   else xp.zeros_like(self.state.u))
        res_div = (self.state.res_div_field
                   if self.state.res_div_field is not None
                   else xp.zeros_like(self.state.u))
        return SolveResult(
            u=asnumpy(self.state.u).astype(np.float64),
            v=asnumpy(self.state.v).astype(np.float64),
            p=asnumpy(self.state.p).astype(np.float64),
            vorticity=asnumpy(vorticity(self.state.u, self.state.v, self.dx, self.dy, self.boundaries)),
            res_mom_field=asnumpy(res_mom),
            res_div_field=asnumpy(res_div),
            mask=asnumpy(self.chi).copy(),
            mesh=self.mesh,
            residuals=list(self.state.residuals),
            cont_residuals=list(self.state.cont_residuals),
            iters=int(self.state.step),
            converged=bool(converged),
            cancelled=bool(self.cancelled),
            elapsed_s=float(elapsed),
            mode=self.cfg.mode,
            frames_u=self.state.frames_u,
            frames_v=self.state.frames_v,
            frames_p=self.state.frames_p,
            frame_times=self.state.frame_times,
        )

    # ---- loops -------------------------------------------------------------
    def _step(self, dt: float, alpha_p: float):
        u_old = self.state.u
        v_old = self.state.v
        p_old = self.state.p

        u_star, v_star, a_P = predictor(
            u_old, v_old, p_old, self.chi,
            dt, self.dx, self.dy, self.nu, self.cfg.eta, self.boundaries,
            force_x=self.force_x, force_y=self.force_y,
        )
        u_new, v_new, p_new, p_prime, info = project(
            u_star, v_star, p_old, a_P, dt, self.dx, self.dy, self.boundaries,
            mg_tol=self.cfg.mg_tol, mg_max=self.cfg.mg_max_cycles, alpha_p=alpha_p,
            p_prime_prev=self._p_prime_prev,
        )
        self._p_prime_prev = p_prime
        r_mom = momentum_residual(u_new, u_old, v_new, v_old, dt)

        self.state.u = u_new
        self.state.v = v_new
        self.state.p = p_new
        self.state.residuals.append(r_mom)
        r_div = info["div_star_max"]
        self.state.cont_residuals.append(r_div)
        self.state.res_mom_field = xp.hypot(u_new - u_old, v_new - v_old) / max(dt, 1e-30)
        self.state.res_div_field = xp.abs(info["div_star"])
        self.state.step += 1
        if self.residual_log is not None:
            self.residual_log(self.state.step, r_mom, r_div)
        return r_mom, info

    def _loop_steady(self, max_iters: int, res_drop: float, alpha_p: float) -> bool:
        window = deque(maxlen=self.cfg.conv_window)
        r0 = None
        r_mom = float("nan")
        for k in range(max_iters):
            if self.should_stop():
                self.cancelled = True
                self.log(f"steady cancelled @ {k} iters (last R_mom={r_mom:.3e})")
                return False
            dt = self._dt_steady(self.state.u, self.state.v)
            r_mom, info = self._step(dt, alpha_p)
            if r0 is None and k >= 10:
                r0 = max(r_mom, 1e-30)
            window.append(r_mom)

            if self.progress is not None and (k % 20 == 0):
                self.progress({
                    "iter": k,
                    "residual": float(r_mom),
                    "r0": float(r0) if r0 is not None else None,
                    "phase": self.phase,
                    "phase_max_iters": int(self._phase_max_iters),
                    "phase_res_drop": float(self._phase_res_drop),
                    "t": None, "t_end": None, "t_buffer": None,
                })
            if k % self.cfg.log_interval == 0:
                self.log(
                    f"[steady {k:6d}] dt={dt:.3e}  R_mom={r_mom:.3e}  "
                    f"div*={info['div_star_max']:.2e}  mg={info['mg_cycles']:d}"
                )

            if r0 is not None and r_mom < res_drop * r0 and flattened(list(window)):
                self.log(f"steady converged @ {k} iters  R_mom={r_mom:.3e} / R0={r0:.3e}")
                return True
        self.log(f"steady did not converge within {max_iters} iters (last R_mom={r_mom:.3e})")
        return False

    def _loop_transient(self):
        t_end = self.cfg.t_end
        frame_dt = self.cfg.frame_dt
        t_buffer = self.cfg.t_buffer
        next_frame_t = t_buffer
        k = 0
        while self.state.t < t_end:
            if self.should_stop():
                self.cancelled = True
                self.log(f"transient cancelled @ t={self.state.t:.3f} "
                         f"(frames captured: {len(self.state.frames_u)})")
                return
            dt = self._dt_transient(self.state.u, self.state.v)
            if self.state.t + dt > t_end:
                dt = t_end - self.state.t
            r_mom, info = self._step(dt, alpha_p=1.0)
            self.state.t += dt
            k += 1

            if self.progress is not None and (k % 20 == 0):
                self.progress({
                    "iter": k,
                    "residual": float(r_mom),
                    "r0": None,
                    "phase": "transient",
                    "phase_max_iters": None,
                    "phase_res_drop": None,
                    "t": float(self.state.t),
                    "t_end": float(t_end),
                    "t_buffer": float(t_buffer),
                })
            if k % self.cfg.log_interval == 0:
                self.log(
                    f"[trans  t={self.state.t:7.3f}] dt={dt:.3e}  R_mom={r_mom:.3e}  "
                    f"div*={info['div_star_max']:.2e}"
                )

            if self.state.t >= next_frame_t and self.state.t >= t_buffer:
                self.state.frames_u.append(asnumpy(self.state.u).astype(np.float32).copy())
                self.state.frames_v.append(asnumpy(self.state.v).astype(np.float32).copy())
                self.state.frames_p.append(asnumpy(self.state.p).astype(np.float32).copy())
                self.state.frame_times.append(float(self.state.t))
                next_frame_t += frame_dt
