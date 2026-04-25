"""Transient-mode smoke test: verify the 2nd-flow-through sampling works.

The transient path integrates in physical time for two flow-through times and
records N_FRAMES snapshots of the second pass. We check the obvious
invariants: frame count/shape, finiteness, actual evolution between frames,
inlet freestream preserved, and sample times in [T_pass, 2*T_pass).
"""
import numpy as np

from solver import config, mesh as mesh_mod
from solver.solver import Solver


BCS = {
    "inlet":  {"type": "inlet_subsonic", "mach": 0.3, "p": 101325.0, "T": 300.0},
    "outlet": {"type": "outlet_subsonic", "p": 101325.0},
    "top":    {"type": "slip_wall"},
    "bottom": {"type": "slip_wall"},
}


def _circle_case():
    m = mesh_mod.build_mesh(width=2.0, height=1.0, nx=120, ny=60)
    cx, cy, r = 0.7, 0.5, 0.12
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    poly = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in theta]
    solid = mesh_mod.rasterize_polygon(poly, m)
    return m, solid, poly


def test_transient_frames_shape_and_finite():
    m, solid, poly = _circle_case()
    s = Solver(m, solid, BCS, polygon_xy=poly, mode="transient")
    r = s.solve()

    N = config.TRANSIENT_FRAMES
    assert r.frames_mach.shape == (N, m["ny"], m["nx"])
    assert r.frames_u.shape == (N, m["ny"], m["nx"])
    assert r.frames_v.shape == (N, m["ny"], m["nx"])
    assert np.all(np.isfinite(r.frames_mach))
    assert np.all(np.isfinite(r.frames_u))
    assert np.all(np.isfinite(r.frames_v))


def test_transient_frames_evolve():
    """Something must actually move between frames — otherwise the animation
    is a static picture and we've lost the point of transient mode."""
    m, solid, poly = _circle_case()
    s = Solver(m, solid, BCS, polygon_xy=poly, mode="transient")
    r = s.solve()

    first = r.frames_mach[0]
    last = r.frames_mach[-1]
    fluid = ~r.mask
    diff = np.linalg.norm((last - first)[fluid])
    base = np.linalg.norm(first[fluid])
    assert diff / base > 5e-3, f"frames are too similar: rel L2 diff = {diff / base:.2e}"


def test_transient_inlet_freestream_preserved():
    """Subsonic inlets pin p and T and let u float with incoming characteristics,
    so acoustic reflections produce a few-percent ripple around the target Mach.
    The meaningful invariant is the time-and-column average, not per-frame."""
    m, solid, poly = _circle_case()
    s = Solver(m, solid, BCS, polygon_xy=poly, mode="transient")
    r = s.solve()

    M_ref = BCS["inlet"]["mach"]
    inlet_mach = r.frames_mach[:, :, 0]
    mean_M = float(inlet_mach.mean())
    assert abs(mean_M - M_ref) / M_ref < 0.03, (
        f"inlet Mach mean drifted: {mean_M:.4f} vs ref {M_ref:.4f}"
    )
    # Per-frame ripple should still be bounded — anything beyond ~15% is a bug.
    assert (inlet_mach.max() - inlet_mach.min()) / M_ref < 0.15


def test_transient_frame_times_in_second_pass():
    """Sample times must lie in the 2nd flow-through window."""
    m, solid, poly = _circle_case()
    s = Solver(m, solid, BCS, polygon_xy=poly, mode="transient")
    r = s.solve()

    gamma, R = s.gamma, s.R
    p = BCS["inlet"]["p"]
    T = BCS["inlet"]["T"]
    rho = p / (R * T)
    c = np.sqrt(gamma * p / rho)
    U = BCS["inlet"]["mach"] * c
    width = m["nx"] * m["dx"]
    T_pass = width / U

    times = r.frame_times
    assert times[0] >= T_pass, f"first sample {times[0]:.4f} < T_pass {T_pass:.4f}"
    assert times[-1] < 2.0 * T_pass + 1e-3
    assert np.all(np.diff(times) >= 0), "frame times are not monotonic"
