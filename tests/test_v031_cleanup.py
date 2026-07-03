from __future__ import annotations

import warnings

from wavesleuth_devito.challenge import collect_leaderboard
from wavesleuth_devito.io import save_json
from wavesleuth_devito.scoring import probability_map_from_mismatch_map
from wavesleuth_devito.uncertainty import candidate_probabilities
from wavesleuth_devito.visualization import visualize_uncertainty
from wavesleuth_devito.world import make_default_world


def _minimal_reconstruction() -> dict:
    world = make_default_world("circle")
    anomaly = world["medium"]["anomaly"]
    candidates = [
        {
            "center_x": float(anomaly["center_x"]),
            "center_z": float(anomaly["center_z"]),
            "radius": float(anomaly["radius"]),
            "anomaly_velocity": 2.2,
            "mismatch": 0.0,
        }
    ]
    reconstruction = {
        "world": world,
        "true_center": {
            "center_x": float(anomaly["center_x"]),
            "center_z": float(anomaly["center_z"]),
            "radius": float(anomaly["radius"]),
        },
        "best_candidate": candidates[0],
        "candidates": candidates,
    }
    reconstruction["uncertainty"] = candidate_probabilities(reconstruction)
    return reconstruction


def test_candidate_probabilities_include_effective_counts() -> None:
    reconstruction = {
        "candidates": [
            {"center_x": 0.1, "center_z": 0.2, "mismatch": 0.0},
            {"center_x": 0.3, "center_z": 0.4, "mismatch": 1.0},
            {"center_x": 0.5, "center_z": 0.6, "mismatch": 2.0},
        ]
    }
    summary = candidate_probabilities(reconstruction)
    assert summary["effective_candidates"] >= 1.0
    assert summary["effective_candidates"] <= 3.0
    assert summary["center_effective_candidates"] >= 1.0
    assert summary["top_3_probability_mass"] <= 1.0


def test_probability_map_summary_include_effective_counts() -> None:
    _prob, summary = probability_map_from_mismatch_map([[0.0, 1.0], [2.0, None]])
    assert summary["effective_candidates"] >= 1.0
    assert summary["inverse_participation_effective_candidates"] >= 1.0


def test_visualize_uncertainty_singleton_grid_without_identical_limit_warning(tmp_path) -> None:
    reconstruction = _minimal_reconstruction()
    out = tmp_path / "uncertainty.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        visualize_uncertainty(reconstruction, out)
    assert out.exists()
    messages = [str(item.message) for item in caught]
    assert not any("Attempting to set identical low and high" in message for message in messages)


def test_leaderboard_rounds_scores_and_includes_best_candidate(tmp_path) -> None:
    def write_summary(dirname: str, challenge: str, score: float) -> None:
        path = tmp_path / dirname
        path.mkdir()
        save_json(
            {
                "challenge": challenge,
                "difficulty": "easy",
                "experimental": False,
                "score": {"iou": 0.8, "center_error": 0.020396, "normalized_center_error": 0.0144, "radius_error": 0.0},
                "challenge_score": {"supported": True, "score": score, "n_forward_runs": 50},
                "runtime_seconds": 1.200123,
                "best_candidate": {"center_x": 0.546, "center_z": 0.5, "radius": 0.12, "anomaly_velocity": 2.2, "mismatch": 0.01},
            },
            path / "challenge_summary.json",
        )

    write_summary("easy", "circle-easy", 71.94954223860289)
    write_summary("noisy", "circle-noisy", 71.94955126226299)
    rows = collect_leaderboard([tmp_path / "easy", tmp_path / "noisy"])
    assert rows[0]["challenge"] == "circle-easy"
    assert rows[0]["score"] == 71.95
    assert rows[0]["best_candidate"]["center_x"] == 0.546
    assert rows[0]["runtime_seconds"] == 1.2
