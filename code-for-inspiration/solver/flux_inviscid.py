"""Roe's approximate Riemann solver with Harten entropy fix, for 2D Euler."""
import numpy as np

from . import config


def euler_flux_x(W, gamma=config.GAMMA):
    rho, u, v, p = W[0], W[1], W[2], W[3]
    E = p / ((gamma - 1.0) * rho) + 0.5 * (u * u + v * v)
    return np.stack(
        [rho * u, rho * u * u + p, rho * u * v, u * (rho * E + p)],
        axis=0,
    )


def _entropy_fix(lam, delta):
    alam = np.abs(lam)
    return np.where(alam < delta, 0.5 * (lam * lam + delta * delta) / np.maximum(delta, 1e-30), alam)


def roe_flux_x(WL, WR, gamma=config.GAMMA, entropy_frac=config.ROE_DELTA_FRAC):
    """Roe flux at x-normal faces. WL, WR are primitive arrays shape (4, ...)."""
    rhoL, uL, vL, pL = WL[0], WL[1], WL[2], WL[3]
    rhoR, uR, vR, pR = WR[0], WR[1], WR[2], WR[3]

    HL = gamma / (gamma - 1.0) * pL / rhoL + 0.5 * (uL * uL + vL * vL)
    HR = gamma / (gamma - 1.0) * pR / rhoR + 0.5 * (uR * uR + vR * vR)

    sL = np.sqrt(rhoL)
    sR = np.sqrt(rhoR)
    denom = sL + sR
    rho_bar = sL * sR
    u_bar = (sL * uL + sR * uR) / denom
    v_bar = (sL * vL + sR * vR) / denom
    H_bar = (sL * HL + sR * HR) / denom
    c2 = (gamma - 1.0) * (H_bar - 0.5 * (u_bar * u_bar + v_bar * v_bar))
    c_bar = np.sqrt(np.maximum(c2, 1e-12))

    drho = rhoR - rhoL
    du = uR - uL
    dv = vR - vL
    dp = pR - pL

    alpha1 = 0.5 * (dp - rho_bar * c_bar * du) / (c_bar * c_bar)  # u-c
    alpha2 = drho - dp / (c_bar * c_bar)                          # u (entropy)
    alpha3 = rho_bar * dv                                         # u (shear)
    alpha5 = 0.5 * (dp + rho_bar * c_bar * du) / (c_bar * c_bar)  # u+c

    delta = entropy_frac * c_bar
    a1 = _entropy_fix(u_bar - c_bar, delta)
    a2 = _entropy_fix(u_bar, delta)
    a3 = a2
    a5 = _entropy_fix(u_bar + c_bar, delta)

    # Dissipation = sum |lam_k| * alpha_k * r_k
    # r1 = [1, u-c,     v,       H - u c]
    # r2 = [1, u,       v,       0.5(u^2+v^2)]
    # r3 = [0, 0,       1,       v]
    # r5 = [1, u+c,     v,       H + u c]
    d0 = a1 * alpha1 + a2 * alpha2 + a5 * alpha5
    d1 = a1 * alpha1 * (u_bar - c_bar) + a2 * alpha2 * u_bar + a5 * alpha5 * (u_bar + c_bar)
    d2 = (a1 * alpha1 + a2 * alpha2 + a5 * alpha5) * v_bar + a3 * alpha3
    d3 = (
        a1 * alpha1 * (H_bar - u_bar * c_bar)
        + a2 * alpha2 * 0.5 * (u_bar * u_bar + v_bar * v_bar)
        + a3 * alpha3 * v_bar
        + a5 * alpha5 * (H_bar + u_bar * c_bar)
    )
    dissipation = np.stack([d0, d1, d2, d3], axis=0)

    FL = euler_flux_x(WL, gamma)
    FR = euler_flux_x(WR, gamma)
    return 0.5 * (FL + FR) - 0.5 * dissipation


def roe_flux_y(WL, WR, gamma=config.GAMMA, entropy_frac=config.ROE_DELTA_FRAC):
    """Roe flux at y-normal faces, computed by rotating (u,v) -> (v,u)."""
    WL_r = np.stack([WL[0], WL[2], WL[1], WL[3]], axis=0)
    WR_r = np.stack([WR[0], WR[2], WR[1], WR[3]], axis=0)
    F_r = roe_flux_x(WL_r, WR_r, gamma, entropy_frac)
    return np.stack([F_r[0], F_r[2], F_r[1], F_r[3]], axis=0)


def wall_flux_x(p):
    """Pressure-only inviscid flux at an x-normal slip wall, given fluid-side pressure."""
    zeros = np.zeros_like(p)
    return np.stack([zeros, p, zeros, zeros], axis=0)


def wall_flux_y(p):
    zeros = np.zeros_like(p)
    return np.stack([zeros, zeros, p, zeros], axis=0)
