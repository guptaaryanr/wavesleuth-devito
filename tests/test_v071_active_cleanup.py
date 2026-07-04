from __future__ import annotations

import numpy as np

from wavesleuth_devito.active import (
    _run_trace_metadata,
    active_score_from_summary,
    collect_active_leaderboard,
    uncertainty_metrics_from_reconstruction,
)
from wavesleuth_devito.io import save_json


def test_run_trace_metadata_3d(tmp_path) -> None:
    path = tmp_path / "run.npz"
    np.savez_compressed(path, receiver_traces=np.zeros((1, 5, 2), dtype=np.float32))
    meta = _run_trace_metadata(path)
    assert meta["layout"] == "shot_time_receiver"
    assert meta["shape"] == [1, 5, 2]
    assert meta["active_trace_shape_standard"] is True


def test_uncertainty_metrics_from_reconstruction() -> None:
    recon = {
        "candidates": [
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 1.0},
            {"center_x": 0.6, "center_z": 0.5, "mismatch": 1.2},
            {"center_x": 0.5, "center_z": 0.5, "mismatch": 1.1},
        ]
    }
    metrics = uncertainty_metrics_from_reconstruction(recon)
    assert metrics["supported"] is True
    assert metrics["n_candidates"] == 3
    assert metrics["n_centers"] == 2
    assert metrics["duplicate_center_candidates"] == 1
    assert metrics["center_effective_candidates"] > 0.0


def test_active_score_and_leaderboard(tmp_path) -> None:
    a = tmp_path / "active_a"
    b = tmp_path / "active_b"
    a.mkdir()
    b.mkdir()
    summary_a = {
        "kind": "circle",
        "strategy": "uncertainty",
        "pool_preset": "boundary-8",
        "round_count": 3,
        "source_history": [{"x": 0.1, "z": 0.1}, {"x": 0.9, "z": 0.9}, {"x": 0.9, "z": 0.1}],
        "runtime_seconds": 1.0,
        "score_delta": 0.4,
        "final_physical_score": {"iou": 0.7, "center_error": 0.05, "normalized_center_error": 0.03},
        "final_uncertainty_summary": {"center_effective_candidates": 8.0, "center_top_probability": 0.3},
        "rounds": [{"physical_score": {"iou": 0.3}}],
    }
    summary_b = {
        **summary_a,
        "strategy": "spread",
        "final_physical_score": {"iou": 0.5, "center_error": 0.08, "normalized_center_error": 0.05},
    }
    save_json(summary_a, a / "active_summary.json")
    save_json(summary_b, b / "active_summary.json")
    leaderboard = collect_active_leaderboard([a, b])["active_leaderboard"]
    assert leaderboard[0]["strategy"] == "uncertainty"
    assert leaderboard[0]["score"] > leaderboard[1]["score"]
    assert active_score_from_summary(summary_a) > active_score_from_summary(summary_b)
