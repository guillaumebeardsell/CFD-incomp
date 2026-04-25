"""MUSCL reconstruction with minmod limiter (2nd-order in space)."""
import numpy as np


def minmod(a, b):
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def muscl_faces(W, axis):
    """Compute left/right primitive states at every interior face along `axis`.

    W has shape (nvar, ny+2g, nx+2g) where g >= 2 ghost layers exist along `axis`.
    Returns WL, WR with the face dimension having length N-3 along `axis`,
    where N is the size of W along `axis`.

    Indexing convention: output face k corresponds to the face between cells
    k+1 and k+2 in the input (i.e. we drop one cell on each end of the valid
    slope region to form face states).
    """
    W = np.moveaxis(W, axis, -1)
    dW_left = W[..., 1:-1] - W[..., :-2]    # slope contribution from the left  (cells 1..N-2)
    dW_right = W[..., 2:] - W[..., 1:-1]    # slope contribution from the right (cells 1..N-2)
    slope = minmod(dW_left, dW_right)        # slope at cells 1..N-2
    WL = W[..., 1:-2] + 0.5 * slope[..., :-1]   # face states using cell on the left
    WR = W[..., 2:-1] - 0.5 * slope[..., 1:]    # face states using cell on the right
    WL = np.moveaxis(WL, -1, axis)
    WR = np.moveaxis(WR, -1, axis)
    return WL, WR
