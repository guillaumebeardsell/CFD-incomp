from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class FieldState:
    u: np.ndarray                  # (ny, nx)
    v: np.ndarray                  # (ny, nx)
    p: np.ndarray                  # (ny, nx)
    chi: np.ndarray                # bool (ny, nx), True in solid
    residuals: list = field(default_factory=list)
    cont_residuals: list = field(default_factory=list)
    res_mom_field: np.ndarray = None   # (ny, nx) per-cell |Δu|/dt from last step
    res_div_field: np.ndarray = None   # (ny, nx) per-cell |div*| from last step
    t: float = 0.0
    step: int = 0
    frames_u: list = field(default_factory=list)
    frames_v: list = field(default_factory=list)
    frames_p: list = field(default_factory=list)
    frame_times: list = field(default_factory=list)

    @classmethod
    def zeros(cls, mesh: dict, chi: np.ndarray, u_init: float = 0.0) -> "FieldState":
        ny, nx = mesh["ny"], mesh["nx"]
        u = np.full((ny, nx), u_init, dtype=np.float64)
        chi_b = np.asarray(chi, dtype=bool)
        u[chi_b] = 0.0
        return cls(
            u=u,
            v=np.zeros((ny, nx), dtype=np.float64),
            p=np.zeros((ny, nx), dtype=np.float64),
            chi=chi_b,
        )
