"""Conservative <-> primitive variable conversions for ideal-gas 2D Euler/NS.

State vectors are stored as (4, ...) numpy arrays with the ordering:
    conservative U = [rho, rho*u, rho*v, rho*E]
    primitive    W = [rho, u,     v,     p]
"""
import numpy as np

from . import config


def primitive_to_conservative(W, gamma=config.GAMMA):
    rho, u, v, p = W[0], W[1], W[2], W[3]
    E = p / ((gamma - 1.0) * rho) + 0.5 * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, rho * E], axis=0)


def conservative_to_primitive(U, gamma=config.GAMMA):
    rho = np.maximum(U[0], config.FLOOR_RHO)
    u = U[1] / rho
    v = U[2] / rho
    E = U[3] / rho
    p = (gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v))
    p = np.maximum(p, config.FLOOR_P)
    return np.stack([rho, u, v, p], axis=0)


def sound_speed(rho, p, gamma=config.GAMMA):
    return np.sqrt(gamma * p / rho)


def total_enthalpy(W, gamma=config.GAMMA):
    rho, u, v, p = W[0], W[1], W[2], W[3]
    E = p / ((gamma - 1.0) * rho) + 0.5 * (u * u + v * v)
    return E + p / rho


def inlet_state(mach, p, T, gamma=config.GAMMA, R=config.R_GAS):
    """Compute primitive freestream state for subsonic/supersonic inlet."""
    rho = p / (R * T)
    c = np.sqrt(gamma * R * T)
    u = mach * c
    return np.array([rho, u, 0.0, p])
