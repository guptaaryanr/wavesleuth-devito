from __future__ import annotations

from wavesleuth_devito.challenge import collect_leaderboard
from wavesleuth_devito.io import save_json
from wavesleuth_devito.report import generate_html_report
from wavesleuth_devito.scoring import score_circle_reconstruction, score_reconstruction
from wavesleuth_devito.world import make_demo_world


def test_scoring_reports_velocity_error() -> None:
    world = make_demo_world()
    score = score_circle_reconstruction(
        world,
        predicted_center_x=0.55,
        predicted_center_z=0.52,
        predicted_radius=0.12,
        predicted_anomaly_velocity=2.41,
    )
    assert score["supported"] is True
    assert round(score["velocity_error"], 2) == 0.21
    assert score["relative_velocity_error"] > 0.0
    assert score["contrast_error"] > 0.0


def test_score_reconstruction_includes_velocity_error() -> None:
    world = make_demo_world()
    reconstruction = {
        "best_candidate": {
            "center_x": 0.55,
            "center_z": 0.52,
            "radius": 0.12,
            "anomaly_velocity": 2.41,
            "mismatch": 0.1,
        }
    }
    score = score_reconstruction(world, reconstruction)
    assert round(score["velocity_error"], 2) == 0.21
    assert score["predicted_anomaly_velocity"] == 2.41


def test_leaderboard_backfills_velocity_error_from_reconstruction(tmp_path) -> None:
    world = make_demo_world()
    recon = {
        "world": world,
        "world_name": world["name"],
        "true_center": {
            "center_x": 0.55,
            "center_z": 0.52,
            "radius": 0.12,
            "anomaly_velocity": 2.2,
        },
        "best_candidate": {
            "center_x": 0.55,
            "center_z": 0.52,
            "radius": 0.12,
            "anomaly_velocity": 2.41,
            "mismatch": 0.1,
        },
        "score": {"supported": True, "iou": 1.0, "center_error": 0.0, "normalized_center_error": 0.0, "radius_error": 0.0},
        "candidate_grid": {"forward_runs": 10},
        "candidates": [
            {"center_x": 0.55, "center_z": 0.52, "radius": 0.12, "anomaly_velocity": 2.41, "mismatch": 0.1}
        ],
        "search": {"search_strategy": "staged"},
        "objective": {},
        "notes": [],
    }
    recon_path = tmp_path / "recon.json"
    save_json(recon, recon_path)
    summary = {
        "challenge": "circle-radius-velocity-staged",
        "difficulty": "hard",
        "experimental": False,
        "reconstruction_path": str(recon_path),
        "score": recon["score"],
        "challenge_score": {
            "supported": True,
            "score": 90.0,
            "n_forward_runs": 10,
            "n_sources": 1,
            "n_receivers": 6,
        },
        "best_candidate": recon["best_candidate"],
    }
    summary_path = tmp_path / "challenge_summary.json"
    save_json(summary, summary_path)
    rows = collect_leaderboard([summary_path])
    assert rows[0]["velocity_error"] == 0.21
    assert rows[0]["relative_velocity_error"] > 0.0


def test_report_backfills_velocity_error_and_staged_note(tmp_path) -> None:
    world = make_demo_world()
    recon = {
        "world": world,
        "world_name": world["name"],
        "true_center": {
            "center_x": 0.55,
            "center_z": 0.52,
            "radius": 0.12,
            "anomaly_velocity": 2.2,
        },
        "best_candidate": {
            "center_x": 0.55,
            "center_z": 0.52,
            "radius": 0.12,
            "anomaly_velocity": 2.41,
            "mismatch": 0.1,
        },
        "score": {"supported": True, "iou": 1.0, "center_error": 0.0, "normalized_center_error": 0.0, "radius_error": 0.0},
        "objective": {},
        "search": {"search_strategy": "staged"},
        "candidate_grid": {"forward_runs": 1},
        "candidates": [
            {"center_x": 0.55, "center_z": 0.52, "radius": 0.12, "anomaly_velocity": 2.41, "mismatch": 0.1}
        ],
        "mismatch_map": [[0.1]],
        "notes": [],
    }
    recon_path = tmp_path / "recon.json"
    save_json(recon, recon_path)
    report_path = generate_html_report(recon_path, tmp_path / "report.html")
    html = report_path.read_text(encoding="utf-8")
    assert "velocity_error" in html
    assert "Staged-search note" in html
    assert "center_effective_candidates" in html
