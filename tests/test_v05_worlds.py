from __future__ import annotations

import numpy as np

from wavesleuth_devito.world import (
    anomaly_mask_from_world,
    background_velocity_model_from_world,
    ellipse_parameters,
    make_default_world,
    validate_world,
    velocity_model_from_world,
    world_with_ellipse_candidate,
)


def test_v05_world_kinds_generate_models() -> None:
    for kind in ("ellipse", "ring", "two-circles", "crack", "circle-layered"):
        world = make_default_world(kind)
        validate_world(world)
        model = velocity_model_from_world(world)
        assert model.shape == (world["grid"]["nx"], world["grid"]["nz"])
        assert np.isfinite(model).all()
        assert np.unique(model).size >= 2
        assert anomaly_mask_from_world(world).any()


def test_circle_layered_background_excludes_circle() -> None:
    world = make_default_world("circle-layered")
    full = velocity_model_from_world(world)
    background = background_velocity_model_from_world(world)
    assert full.shape == background.shape
    assert not np.allclose(full, background)
    assert np.unique(background).size >= 2


def test_ellipse_candidate_copy() -> None:
    world = make_default_world("ellipse")
    params = ellipse_parameters(world)
    assert params is not None
    candidate = world_with_ellipse_candidate(
        world,
        center_x=0.5,
        center_z=0.5,
        radius_x=params["radius_x"],
        radius_z=params["radius_z"],
        angle_degrees=params["angle_degrees"],
        anomaly_velocity=2.3,
    )
    assert candidate["medium"]["anomaly"]["kind"] == "ellipse"
    assert candidate["medium"]["anomaly"]["center_x"] == 0.5
    assert candidate["medium"]["anomaly_velocity"] == 2.3
