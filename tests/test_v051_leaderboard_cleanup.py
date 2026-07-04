from __future__ import annotations

from wavesleuth_devito.challenge import collect_leaderboard
from wavesleuth_devito.io import save_json


def test_leaderboard_omits_non_applicable_null_shape_fields(tmp_path) -> None:
    summary = {
        "challenge": "circle-easy",
        "difficulty": "easy",
        "experimental": False,
        "runtime_seconds": 1.23,
        "score": {
            "supported": True,
            "iou": 0.8,
            "center_error": 0.02,
            "normalized_center_error": 0.01,
            "radius_error": 0.0,
            "radius_x_error": None,
            "radius_z_error": None,
            "angle_error_degrees": None,
            "velocity_error": 0.0,
            "relative_velocity_error": 0.0,
        },
        "challenge_score": {
            "supported": True,
            "score": 70.0,
            "n_forward_runs": 10,
            "n_sources": 2,
            "n_receivers": 4,
        },
        "best_candidate": {"center_x": 0.5, "center_z": 0.5, "radius": 0.12, "anomaly_velocity": 2.2},
    }
    path = tmp_path / "challenge_summary.json"
    save_json(summary, path)
    rows = collect_leaderboard([path])
    assert len(rows) == 1
    row = rows[0]
    assert "radius_error" in row
    assert "radius_x_error" not in row
    assert "radius_z_error" not in row
    assert "angle_error_degrees" not in row
    assert row["velocity_error"] == 0.0
