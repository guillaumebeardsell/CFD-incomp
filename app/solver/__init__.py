import os as _os

# The MG convergence check calls `np.linalg.norm` on small (~20k cell) arrays
# every V-cycle. With BLAS multi-threaded, the per-call dispatch overhead
# dwarfs the compute, costing >50% of solver time. Pin BLAS to one thread
# unless the user explicitly overrides; the threaded Numba kernels (when
# enabled) own their own parallelism.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

from .config import SolverConfig, FIDELITY
from .solver import Solver, SolveResult

__all__ = ["SolverConfig", "FIDELITY", "Solver", "SolveResult"]
