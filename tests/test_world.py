from __future__ import annotations

import numpy as np

from wavesleuth_devito.world import SUPPORTED_WORLD_KINDS, make_default_world, validate_world, velocity_model_from_world


def test_default_worlds_validate() -> None:
    for kind in SUPPORTED_WORLD_KINDS:
        world = make_default_world(kind)
        validate_world(world)
        assert world["medium"]["anomaly"]["kind"] == kind


def test_velocity_model_shape_and_values_for_circle() -> None:
    world = make_default_world("circle")
    model = velocity_model_from_world(world)
    assert model.shape == (world["grid"]["nx"], world["grid"]["nz"])
    assert np.isclose(model.min(), world["medium"]["background_velocity"])
    assert np.isclose(model.max(), world["medium"]["anomaly_velocity"])
    assert np.count_nonzero(model == world["medium"]["anomaly_velocity"]) > 0


def test_blob_generation_is_deterministic() -> None:
    a = make_default_world("blobs", seed=777)
    b = make_default_world("blobs", seed=777)
    assert a["medium"]["anomaly"] == b["medium"]["anomaly"]
