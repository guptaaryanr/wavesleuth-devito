from __future__ import annotations

import numpy as np

from wavesleuth_devito.scoring import center_error, iou_score, score_circle_reconstruction
from wavesleuth_devito.world import make_default_world


def test_center_error() -> None:
    assert center_error((0.0, 0.0), (3.0, 4.0)) == 5.0


def test_iou_score() -> None:
    true = np.array([[True, True], [False, False]])
    pred = np.array([[True, False], [True, False]])
    assert np.isclose(iou_score(true, pred), 1.0 / 3.0)


def test_score_circle_perfect_center() -> None:
    world = make_default_world("circle")
    anomaly = world["medium"]["anomaly"]
    score = score_circle_reconstruction(
        world,
        predicted_center_x=anomaly["center_x"],
        predicted_center_z=anomaly["center_z"],
        predicted_radius=anomaly["radius"],
    )
    assert score["supported"] is True
    assert score["center_error"] == 0.0
    assert np.isclose(score["iou"], 1.0)
