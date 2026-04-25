"""Cartesian grid + rasterization of a user-drawn polygon into a solid mask."""
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon


def build_mesh(width, height, nx, ny):
    dx = width / nx
    dy = height / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "x0": 0.0,
        "y0": 0.0,
        "width": width,
        "height": height,
        "xc": xc,
        "yc": yc,
        "X": X,
        "Y": Y,
    }


def rasterize_polygon(polygon_xy, mesh):
    """Return (ny, nx) bool mask, True where a cell center lies inside the polygon."""
    ny, nx = mesh["ny"], mesh["nx"]
    if polygon_xy is None or len(polygon_xy) < 3:
        return np.zeros((ny, nx), dtype=bool)
    poly = Polygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    mask = contains_xy(poly, mesh["X"], mesh["Y"])
    return np.asarray(mask, dtype=bool)


def face_masks(solid):
    """Classify x- and y-faces based on the solid mask.

    Returns
    -------
    xf_type : (ny, nx+1) int8, for each vertical face between cells (j, i-1) and (j, i):
        0 = fluid-fluid, 1 = wall (fluid/solid), 2 = solid-solid
    xf_fluid_side : (ny, nx+1) int8, which side is the fluid when xf_type == 1
        0 = left (i-1), 1 = right (i)
    yf_type, yf_fluid_side : analogous for horizontal faces, shape (ny+1, nx).
    """
    ny, nx = solid.shape
    # x-faces: shape (ny, nx+1). Treat anything outside the domain as fluid (handled by BCs).
    solid_pad_x = np.zeros((ny, nx + 2), dtype=bool)
    solid_pad_x[:, 1:-1] = solid
    left = solid_pad_x[:, :-1]   # (ny, nx+1)
    right = solid_pad_x[:, 1:]   # (ny, nx+1)
    xf_type = np.where(
        left & right, 2,
        np.where(left | right, 1, 0),
    ).astype(np.int8)
    xf_fluid_side = np.where(right & ~left, 0, 1).astype(np.int8)  # fluid on the left when right is solid

    solid_pad_y = np.zeros((ny + 2, nx), dtype=bool)
    solid_pad_y[1:-1, :] = solid
    bot = solid_pad_y[:-1, :]
    top = solid_pad_y[1:, :]
    yf_type = np.where(
        bot & top, 2,
        np.where(bot | top, 1, 0),
    ).astype(np.int8)
    yf_fluid_side = np.where(top & ~bot, 0, 1).astype(np.int8)

    return xf_type, xf_fluid_side, yf_type, yf_fluid_side
