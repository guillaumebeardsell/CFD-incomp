"""Default physical and numerical constants for the solver."""

GAMMA = 1.4
R_GAS = 287.05
MU_REF = 1.716e-5
T_SUTH_REF = 273.15
S_SUTH = 110.4
PR = 0.72

CFL = 0.4
CFL_IRS = 2.5
CFL_IRS_IB = 2.0  # modest reduction when ghost-cell IB is active (reflected wall raises effective wave speed)
EPS_IRS = 1.3
ROE_DELTA_FRAC = 0.1
FLOOR_RHO = 1e-6
FLOOR_P = 1e-3

MAX_ITERS_DEFAULT = 20000
RES_DROP_DEFAULT = 1e-5
LOG_INTERVAL = 200

TRANSIENT_FRAMES = 60
MAX_TRANSIENT_ITERS = 30000

FIDELITY = {
    "coarse": (160, 80),
    "medium": (240, 120),
    "fine":   (360, 180),
}
