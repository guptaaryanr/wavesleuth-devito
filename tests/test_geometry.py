from __future__ import annotations

import numpy as np

from wavesleuth_devito.geometry import candidate_centers, physical_to_grid_index, receiver_coordinates, source_coordinates
from wavesleuth_devito.world import make_default_world


def test_source_receiver_coordinates() -> None:
    world = make_default_world("circle")
    src = source_coordinates(world)
    rec = receiver_coordinates(world)
    assert src.shape == (1, 2)
    assert rec.shape[1] == 2
    assert np.all(rec[:, 1] > src[0, 1])


def test_physical_to_grid_index_bounds() -> None:
    world = make_default_world("circle")
    ix, iz = physical_to_grid_index(world, 0.0, 1.0)
    assert ix == 0
    assert iz == world["grid"]["nz"] - 1


def test_candidate_centers_count() -> None:
    world = make_default_world("circle")
    centers = candidate_centers(world, 5)
    assert len(centers) == 25
    assert all(0.0 <= float(c["center_x"]) <= 1.0 for c in centers)
    assert all(0.0 <= float(c["center_z"]) <= 1.0 for c in centers)
