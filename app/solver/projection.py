import numpy as np

from .operators import gradient, rhie_chow_faces, divergence_from_faces
from . import poisson_mg
from ._xp import xp


def project(u_star, v_star, p_old, a_P, dt, dx, dy, bcs,
            mg_tol=1e-4, mg_max=30, alpha_p=1.0, p_prime_prev=None):
    """Pressure-correction projection step.

    Returns (u_new, v_new, p_new, p_prime, info_dict).
    info_dict = {"mg_cycles", "mg_resid", "div_star_max"}.
    `p_prime_prev` (optional) warm-starts the MG solve with the previous
    step's correction, which typically halves V-cycle count once the flow
    is developed.
    """
    p_old_pad = bcs.pad_pressure(p_old)
    u_f, v_f = rhie_chow_faces(u_star, v_star, p_old_pad, a_P, dt, dx, dy, bcs)

    div_star = divergence_from_faces(u_f, v_f, dx, dy)
    rhs = div_star / dt

    p_prime, n_cycles, resid = poisson_mg.solve(
        rhs, dx, dy, tol=mg_tol, max_cycles=mg_max, p0=p_prime_prev
    )

    p_prime_pad = bcs.pad_pressure_correction(p_prime)
    gppx, gppy = gradient(p_prime_pad, dx, dy)
    u_new = u_star - dt * gppx / a_P
    v_new = v_star - dt * gppy / a_P

    p_new = p_old + alpha_p * p_prime
    return u_new, v_new, p_new, p_prime, {
        "mg_cycles": n_cycles,
        "mg_resid": float(resid),
        "div_star_max": float(np.max(np.abs(div_star))),
        "div_star": div_star,
    }
