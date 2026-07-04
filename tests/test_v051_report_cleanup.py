from __future__ import annotations

from wavesleuth_devito.report import _report_uncertainty_summary


def test_report_backfills_old_duplicate_center_uncertainty() -> None:
    reconstruction = {
        "uncertainty": {
            "center_probabilities": [{"center_x": 0.5, "center_z": 0.5, "probability": 0.9}],
            "effective_candidates": 2.0,
        },
        "candidates": [
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 10.0},
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 10.0},
            {"center_x": 0.55, "center_z": 0.5, "mismatch": 1.0},
        ],
    }
    summary = _report_uncertainty_summary(reconstruction)
    assert summary["center_probability_mode"] == "unique-center-min-mismatch"
    assert summary["center_probabilities"][0]["center_x"] == 0.55
