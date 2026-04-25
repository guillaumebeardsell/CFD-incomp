from dataclasses import dataclass

FIDELITY = {
    "coarse": (200, 100),
    "medium": (320, 160),
    "fine":   (480, 240),
}


@dataclass
class SolverConfig:
    mode: str = "steady"           # "steady" | "transient"
    Re: float = 40.0
    U_ref: float = 1.0
    L_ref: float = 1.0
    eta: float = 1e-3              # Brinkman penalization (smaller = harder solid)
    cfl_steady: float = 0.5
    cfl_transient: float = 0.3
    alpha_p_steady: float = 0.7    # pressure under-relaxation in steady mode
    res_drop: float = 1e-4         # convergence target (R_mom final / initial)
    max_iters: int = 20000
    t_end: float = 120.0
    t_buffer: float = 20.0         # skip this much physical time before recording
    frame_dt: float = 0.5          # frame sampling cadence (physical time)
    log_interval: int = 100
    mg_tol: float = 1e-4
    mg_max_cycles: int = 30
    conv_window: int = 200         # residual flattening window
