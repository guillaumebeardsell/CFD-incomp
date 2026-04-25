import numpy as np

from ._xp import xp


def momentum_residual(u_new, u_old, v_new, v_old, dt) -> float:
    du = u_new - u_old
    dv = v_new - v_old
    mag = float(xp.sqrt(xp.mean(u_new ** 2 + v_new ** 2))) + 1e-30
    return float(xp.sqrt(xp.mean(du ** 2 + dv ** 2))) / (dt * mag)


def flattened(window) -> bool:
    """True if the residual window is flat (max/min ratio < 1.05)."""
    if len(window) < 20:
        return False
    a = np.asarray(window, dtype=np.float64)
    lo = float(a.min())
    hi = float(a.max())
    return hi < 1e-8 or hi / max(lo, 1e-30) < 2.0
