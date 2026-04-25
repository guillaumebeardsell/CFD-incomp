import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.vectorized import contains


def build_mesh(width: float, height: float, nx: int, ny: int) -> dict:
    dx = width / nx
    dy = height / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc)
    return {
        "nx": int(nx), "ny": int(ny),
        "dx": float(dx), "dy": float(dy),
        "x0": 0.0, "y0": 0.0,
        "width": float(width), "height": float(height),
        "xc": xc, "yc": yc, "X": X, "Y": Y,
    }


def rasterize_polygon(polygon_xy, mesh: dict) -> np.ndarray:
    """Rasterize a polygon to a cell-centered bool mask.

    Hand-drawn polygons are often technically invalid (start/end overlap,
    near-duplicate vertices, slight self-intersection at the closing edge).
    We try `buffer(0)` which repairs most such cases; only a truly empty or
    degenerate geometry falls through to an all-False mask.
    """
    ny, nx = mesh["ny"], mesh["nx"]
    if not polygon_xy or len(polygon_xy) < 3:
        return np.zeros((ny, nx), dtype=bool)
    poly = Polygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return np.zeros((ny, nx), dtype=bool)
    if isinstance(poly, MultiPolygon):
        mask = np.zeros((ny, nx), dtype=bool)
        for sub in poly.geoms:
            mask |= contains(sub, mesh["X"], mesh["Y"])
        return mask
    return contains(poly, mesh["X"], mesh["Y"])


def rasterize_disk(cx: float, cy: float, r: float, mesh: dict) -> np.ndarray:
    return (mesh["X"] - cx) ** 2 + (mesh["Y"] - cy) ** 2 < r ** 2
