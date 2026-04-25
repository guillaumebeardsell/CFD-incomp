import numpy as np

from solver import mesh as mesh_mod


def test_grid_spacing():
    m = mesh_mod.build_mesh(2.0, 1.0, 20, 10)
    assert m["dx"] == 0.1
    assert m["dy"] == 0.1
    assert m["xc"].shape == (20,)
    assert m["yc"].shape == (10,)
    assert np.isclose(m["xc"][0], 0.05)
    assert np.isclose(m["xc"][-1], 1.95)


def test_rasterize_square():
    m = mesh_mod.build_mesh(1.0, 1.0, 100, 100)
    poly = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    solid = mesh_mod.rasterize_polygon(poly, m)
    # A unit square from 0.25 to 0.75 covers 0.5*0.5 = 0.25 of area.
    area = solid.sum() * m["dx"] * m["dy"]
    assert abs(area - 0.25) < 1e-2
    # Centre of domain must be solid
    assert solid[50, 50]
    # Corner must not be solid
    assert not solid[0, 0]


def test_face_masks_isolated_cell():
    # Single solid cell in a 5x5 domain -> surrounded by 4 wall faces.
    m = mesh_mod.build_mesh(1.0, 1.0, 5, 5)
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 2] = True
    xf_type, _, yf_type, _ = mesh_mod.face_masks(solid)
    # xf_type shape (5, 6): faces left/right of each column
    assert xf_type.shape == (5, 6)
    assert yf_type.shape == (6, 5)
    # Wall faces around the solid cell
    assert xf_type[2, 2] == 1  # face left of solid cell (between col 1 and 2)
    assert xf_type[2, 3] == 1  # face right of solid cell
    assert yf_type[2, 2] == 1  # face below solid cell
    assert yf_type[3, 2] == 1  # face above solid cell
