from __future__ import annotations

from wavesleuth_devito.uncertainty import candidate_probabilities


def test_center_uncertainty_uses_best_unique_center_not_duplicate_sum() -> None:
    reconstruction = {
        "candidates": [
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 10.0, "stage": "coarse"},
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 10.0, "stage": "refine"},
            {"center_x": 0.55, "center_z": 0.5, "mismatch": 1.0, "stage": "refine"},
        ]
    }
    summary = candidate_probabilities(reconstruction, temperature=1.0)
    assert summary["center_probability_mode"] == "unique-center-min-mismatch"
    assert summary["n_candidates"] == 3
    assert summary["n_centers"] == 2
    assert summary["duplicate_center_candidates"] == 1
    assert summary["center_probabilities"][0]["center_x"] == 0.55
    assert summary["center_probabilities"][0]["center_z"] == 0.5
    assert summary["center_probabilities"][0]["mismatch"] == 1.0


def test_center_probability_representative_uses_min_mismatch() -> None:
    reconstruction = {
        "candidates": [
            {"center_x": 0.1, "center_z": 0.2, "mismatch": 5.0},
            {"center_x": 0.1, "center_z": 0.2, "mismatch": 2.0},
            {"center_x": 0.3, "center_z": 0.4, "mismatch": 3.0},
        ]
    }
    summary = candidate_probabilities(reconstruction, temperature=1.0)
    first = summary["center_probabilities"][0]
    assert first["center_x"] == 0.1
    assert first["center_z"] == 0.2
    assert first["mismatch"] == 2.0
