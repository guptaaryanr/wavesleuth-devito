from __future__ import annotations

import numpy as np

from wavesleuth_devito.cellmask import active_cell_tuples, cell_bounds, mask_blocks_mask_from_cells, world_with_mask_blocks_candidate
from wavesleuth_devito.metadata import __version__
from wavesleuth_devito.scoring import score_reconstruction
from wavesleuth_devito.world import SUPPORTED_WORLD_KINDS, make_default_world, velocity_model_from_world


def test_version_is_v080() -> None:
    assert tuple(int(part) for part in __version__.split(".")[:3]) >= (0, 8, 0)


def test_mask_blocks_world_generation_and_model() -> None:
    assert "mask-blocks" in SUPPORTED_WORLD_KINDS
    world = make_default_world("mask-blocks")
    anomaly = world["medium"]["anomaly"]
    assert anomaly["kind"] == "mask-blocks"
    assert len(anomaly["active_cells"]) == 5
    model = velocity_model_from_world(world)
    assert model.shape == (world["grid"]["nx"], world["grid"]["nz"])
    assert np.isclose(model.min(), world["medium"]["background_velocity"])
    assert np.isclose(model.max(), world["medium"]["anomaly_velocity"])


def test_mask_blocks_cell_helpers_and_candidate_world() -> None:
    world = make_default_world("mask-blocks")
    x0, x1, z0, z1 = cell_bounds(world, 2, 2)
    assert x0 < x1
    assert z0 < z1
    mask = mask_blocks_mask_from_cells(world, [{"i": 2, "j": 2}])
    assert mask.any()
    candidate = world_with_mask_blocks_candidate(world, active_cells=[{"i": 2, "j": 2}])
    assert active_cell_tuples(candidate["medium"]["anomaly"]["active_cells"]) == [(2, 2)]


def test_mask_blocks_perfect_score() -> None:
    world = make_default_world("mask-blocks")
    active = world["medium"]["anomaly"]["active_cells"]
    reconstruction = {
        "best_candidate": {
            "kind": "mask-blocks",
            "cell_grid_size": world["medium"]["anomaly"]["cell_grid_size"],
            "active_cells": active,
            "anomaly_velocity": world["medium"]["anomaly_velocity"],
            "mismatch": 0.0,
        }
    }
    score = score_reconstruction(world, reconstruction)
    assert score["supported"] is True
    assert np.isclose(score["iou"], 1.0)
    assert score["cell_count_error"] == 0
