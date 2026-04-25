import numpy as np

from ._xp import xp, asarray


class Boundaries:
    """Holds BC spec for the 4 domain edges; obstacle is enforced by Brinkman.

    Supported types:
      inlet:    "inlet_velocity" with {"speed", "angle_deg"}
      outlet:   "outlet_pressure" with {"p"}
      top/bot:  "slip" | "symmetry" | "no_slip"
    """

    def __init__(self, bcs: dict):
        self.inlet = bcs["inlet"]
        self.outlet = bcs["outlet"]
        self.top = bcs["top"]
        self.bottom = bcs["bottom"]

        t_in = self.inlet.get("type", "inlet_velocity")
        if t_in != "inlet_velocity":
            raise ValueError(f"Unsupported inlet type: {t_in}")
        # `profile` is an optional (ny,) array of per-row u values at the inlet;
        # if set, it overrides `speed/angle_deg`.
        prof = self.inlet.get("profile")
        if prof is not None:
            prof = np.asarray(prof, dtype=np.float64)
            self.U_in = asarray(prof)                # (ny,) array, on backend
            self.V_in = asarray(np.zeros_like(prof))
        else:
            speed = float(self.inlet.get("speed", 1.0))
            ang = float(self.inlet.get("angle_deg", 0.0))
            a = np.deg2rad(ang)
            self.U_in = speed * np.cos(a)
            self.V_in = speed * np.sin(a)

        t_out = self.outlet.get("type", "outlet_pressure")
        if t_out != "outlet_pressure":
            raise ValueError(f"Unsupported outlet type: {t_out}")
        self.P_out = float(self.outlet.get("p", 0.0))

        for side_name, side in (("top", self.top), ("bottom", self.bottom)):
            t = side.get("type", "slip")
            if t not in ("slip", "symmetry", "no_slip"):
                raise ValueError(f"Unsupported {side_name} type: {t}")

    def pad_velocity(self, u, v):
        """Return (u_pad, v_pad), each shape (ny+2, nx+2), ghosts filled per BCs."""
        ny, nx = u.shape
        up = xp.empty((ny + 2, nx + 2), dtype=u.dtype)
        vp = xp.empty((ny + 2, nx + 2), dtype=v.dtype)
        up[1:-1, 1:-1] = u
        vp[1:-1, 1:-1] = v

        # Inlet (west, i=0 ghost column): Dirichlet u=U_in at face => ghost = 2*U_in - u_interior
        # U_in / V_in may be a scalar or a (ny,) per-row profile.
        up[1:-1, 0] = 2.0 * self.U_in - u[:, 0]
        vp[1:-1, 0] = 2.0 * self.V_in - v[:, 0]

        # Outlet (east, i=-1 ghost column): Neumann => ghost = interior
        up[1:-1, -1] = u[:, -1]
        vp[1:-1, -1] = v[:, -1]

        self._fill_wall(up[0, 1:-1], vp[0, 1:-1], u[0, :], v[0, :], self.bottom["type"])
        self._fill_wall(up[-1, 1:-1], vp[-1, 1:-1], u[-1, :], v[-1, :], self.top["type"])

        # Corner ghosts (not used by 5-point stencils, filled for safety)
        up[0, 0] = up[0, 1]; up[0, -1] = up[0, -2]
        up[-1, 0] = up[-1, 1]; up[-1, -1] = up[-1, -2]
        vp[0, 0] = vp[0, 1]; vp[0, -1] = vp[0, -2]
        vp[-1, 0] = vp[-1, 1]; vp[-1, -1] = vp[-1, -2]
        return up, vp

    @staticmethod
    def _fill_wall(up_row, vp_row, u_edge, v_edge, wall_type: str):
        if wall_type in ("slip", "symmetry"):
            up_row[:] = u_edge          # du/dn = 0
            vp_row[:] = -v_edge         # v_wall = 0 via reflection
        elif wall_type == "no_slip":
            up_row[:] = -u_edge
            vp_row[:] = -v_edge
        else:
            raise ValueError(wall_type)

    def pad_pressure(self, p):
        """Ghosts for p^n: outlet Dirichlet p=P_out, others Neumann."""
        ny, nx = p.shape
        pp = xp.empty((ny + 2, nx + 2), dtype=p.dtype)
        pp[1:-1, 1:-1] = p
        pp[1:-1, 0] = p[:, 0]                              # inlet Neumann
        pp[1:-1, -1] = 2.0 * self.P_out - p[:, -1]         # outlet Dirichlet
        pp[0, 1:-1] = p[0, :]                              # bottom Neumann
        pp[-1, 1:-1] = p[-1, :]                            # top Neumann
        pp[0, 0] = pp[0, 1]; pp[0, -1] = pp[0, -2]
        pp[-1, 0] = pp[-1, 1]; pp[-1, -1] = pp[-1, -2]
        return pp

    def pad_pressure_correction(self, pp_cell):
        """Ghosts for p': outlet Dirichlet p'=0, others Neumann."""
        ny, nx = pp_cell.shape
        out = xp.empty((ny + 2, nx + 2), dtype=pp_cell.dtype)
        out[1:-1, 1:-1] = pp_cell
        out[1:-1, 0] = pp_cell[:, 0]
        out[1:-1, -1] = -pp_cell[:, -1]
        out[0, 1:-1] = pp_cell[0, :]
        out[-1, 1:-1] = pp_cell[-1, :]
        out[0, 0] = out[0, 1]; out[0, -1] = out[0, -2]
        out[-1, 0] = out[-1, 1]; out[-1, -1] = out[-1, -2]
        return out
