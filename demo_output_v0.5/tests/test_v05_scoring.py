from __future__ import annotations

import numpy as np

from wavesleuth_devito.scoring import score_reconstruction
from wavesleuth_devito.world import make_default_world


def test_score_ellipse_perfect_candidate() -> None:
    world = make_default_world("ellipse")
    anomaly = world["medium"]["anomaly"]
    reconstruction = {
        "best_candidate": {
            "kind": "ellipse",
            "center_x": anomaly["center_x"],
            "center_z": anomaly["center_z"],
            "radius_x": anomaly["radius_x"],
            "radius_z": anomaly["radius_z"],
            "angle_degrees": anomaly["angle_degrees"],
            "anomaly_velocity": world["medium"]["anomaly_velocity"],
            "mismatch": 0.0,
        }
    }
    score = score_reconstruction(world, reconstruction)
    assert score["supported"] is True
    assert score["target_kind"] == "ellipse"
    assert score["center_error"] == 0.0
    assert score["radius_x_error"] == 0.0
    assert score["radius_z_error"] == 0.0
    assert score["angle_error_degrees"] == 0.0
    assert np.isclose(score["iou"], 1.0)
    assert score["velocity_error"] == 0.0


def test_score_non_circle_non_ellipse_is_explicitly_unsupported() -> None:
    world = make_default_world("ring")
    reconstruction = {"best_candidate": {"center_x": 0.5, "center_z": 0.5}}
    score = score_reconstruction(world, reconstruction)
    assert score["supported"] is False
    assert "not implemented" in score["message"]
