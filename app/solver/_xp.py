"""Backend dispatch for NumPy / CuPy.

Set `CFD_BACKEND=cupy` to route field allocations and the four hot kernels
through CuPy on a CUDA device. Default is NumPy, in which case `xp` is
literally numpy and `asarray`/`asnumpy` are no-ops.

`xp` is the module to use for array creation and elementwise ops in code
that needs to run on either backend (e.g. ghost-cell padding, MG transfer
operators). The four numerically-heavy kernels are dispatched separately
in `app/solver/_jit.py`.
"""
from __future__ import annotations

import os

_BACKEND = os.environ.get("CFD_BACKEND", "numpy").lower()
USE_GPU = _BACKEND == "cupy"

if USE_GPU:
    import cupy as xp  # type: ignore[import-not-found]

    def asarray(a):
        return xp.asarray(a)

    def asnumpy(a):
        return xp.asnumpy(a)
else:
    import numpy as xp  # type: ignore[no-redef]

    def asarray(a):
        return a

    def asnumpy(a):
        return a
