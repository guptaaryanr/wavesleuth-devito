from __future__ import annotations

import json
from pathlib import Path

from wavesleuth_devito.challenge import clean_challenge_output
from wavesleuth_devito.io import save_json
from wavesleuth_devito.report import generate_html_report
from wavesleuth_devito.scoring import budgeted_challenge_score


def test_budgeted_challenge_score_ignores_runtime() -> None:
    reconstruction_score = {"supported": True, "iou": 0.8, "normalized_center_error": 0.02}
    fast = budgeted_challenge_score(
        reconstruction_score,
        n_forward_runs=50,
        n_sources=4,
        n_receivers=12,
        runtime_seconds=1.0,
    )
    slow = budgeted_challenge_score(
        reconstruction_score,
        n_forward_runs=50,
        n_sources=4,
        n_receivers=12,
        runtime_seconds=999.0,
    )
    assert fast["score"] == slow["score"]
    assert fast["runtime_scored"] is False
    assert "seconds" not in fast["formula"]


def test_clean_challenge_output_removes_only_known_artifacts(tmp_path: Path) -> None:
    stale_world = tmp_path / "worlds" / "circle-noisy.json"
    stale_run = tmp_path / "runs" / "circle-noisy_obs.npz"
    stale_recon = tmp_path / "runs" / "circle-noisy_recon.json"
    stale_report_asset = tmp_path / "reports" / "report_assets" / "old.png"
    keep = tmp_path / "notes.txt"
    for path in (stale_world, stale_run, stale_recon, stale_report_asset, keep):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (tmp_path / "challenge_summary.json").write_text("{}", encoding="utf-8")

    removed = clean_challenge_output(tmp_path)

    assert "worlds/circle-noisy.json" in removed
    assert "runs/circle-noisy_obs.npz" in removed
    assert "runs/circle-noisy_recon.json" in removed
    assert "reports/report_assets" in removed
    assert not stale_world.exists()
    assert not stale_run.exists()
    assert not stale_recon.exists()
    assert not stale_report_asset.exists()
    assert keep.exists()


def test_report_backfills_uncertainty_for_old_reconstruction(tmp_path: Path) -> None:
    world = {
        "name": "old_recon_world",
        "grid": {"nx": 20, "nz": 20, "extent_x": 1.0, "extent_z": 1.0},
        "medium": {
            "background_velocity": 1.5,
            "anomaly_velocity": 2.2,
            "anomaly": {"kind": "circle", "center_x": 0.5, "center_z": 0.5, "radius": 0.1},
        },
        "acquisition": {
            "sources": [{"x": 0.2, "z": 0.2}],
            "receivers": [{"x": 0.3, "z": 0.8}, {"x": 0.7, "z": 0.8}],
        },
        "simulation": {"nt": 20, "dt": 0.001, "space_order": 2, "source_frequency": 20.0},
    }
    reconstruction = {
        "world_name": "old_recon_world",
        "world": world,
        "score": {"supported": True, "iou": 0.5},
        "best_candidate": {"center_x": 0.5, "center_z": 0.5, "radius": 0.1, "mismatch": 0.0},
        "candidate_grid": {"xs": [0.5], "zs": [0.5]},
        "mismatch_map": [[0.0]],
        "candidates": [
            {"center_x": 0.5, "center_z": 0.5, "radius": 0.1, "anomaly_velocity": 2.2, "mismatch": 0.0},
            {"center_x": 0.4, "center_z": 0.5, "radius": 0.1, "anomaly_velocity": 2.2, "mismatch": 1.0},
        ],
        "uncertainty": {},
        "notes": [],
    }
    recon_path = tmp_path / "recon.json"
    save_json(reconstruction, recon_path)
    report_path = generate_html_report(recon_path, tmp_path / "report.html")
    html = report_path.read_text(encoding="utf-8")
    assert "effective_candidates" in html
    assert "center_effective_candidates" in html



def test_leaderboard_recomputes_old_runtime_scored_summary(tmp_path: Path) -> None:
    from wavesleuth_devito.challenge import collect_leaderboard

    summary = {
        "challenge": "circle-easy",
        "difficulty": "easy",
        "experimental": False,
        "score": {
            "supported": True,
            "iou": 0.8,
            "center_error": 0.02,
            "normalized_center_error": 0.014,
            "radius_error": 0.0,
        },
        "challenge_score": {
            "supported": True,
            "score": 71.0,
            "n_forward_runs": 50,
            "n_sources": 4,
            "n_receivers": 12,
            "runtime_seconds": 999.0,
            "formula": "old formula with seconds",
        },
        "runtime_seconds": 999.0,
        "best_candidate": {"center_x": 0.5, "center_z": 0.5, "radius": 0.12, "anomaly_velocity": 2.2},
    }
    save_json(summary, tmp_path / "challenge_summary.json")
    rows = collect_leaderboard([tmp_path])
    expected = 100.0 * 0.8 - 20.0 * 0.014 - 0.08 * 50 - 0.75 * 4 - 0.15 * 12
    assert rows[0]["score"] == round(expected, 3)
