from __future__ import annotations

import json

import numpy as np

from wavesleuth_devito import __version__
from wavesleuth_devito.blind import is_blind_public_world, public_world_from_secret
from wavesleuth_devito.io import save_run_npz
from wavesleuth_devito.release import generate_release_html_report, validate_active_directory
from wavesleuth_devito.world import background_velocity_model_from_world, make_default_world


def _version_tuple(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split(".")[:3])  # type: ignore[return-value]


def test_version_is_at_least_v091() -> None:
    assert _version_tuple(__version__) >= (0, 9, 1)


def test_validate_active_directory_does_not_duplicate_kind_kwarg(tmp_path) -> None:
    active_dir = tmp_path / "active_demo"
    active_dir.mkdir()
    (active_dir / "active_summary.json").write_text(
        json.dumps(
            {
                "kind": "circle",
                "strategy": "uncertainty",
                "initial_reconstruction_score": 0.0,
                "final_reconstruction_score": 0.5,
                "score_delta": 0.5,
                "rounds": [{"round": 1, "trace_shape": [1, 10, 2]}],
            }
        ),
        encoding="utf-8",
    )
    result = validate_active_directory(active_dir)
    assert result["ok"] is True
    assert result["kind"] == "active"
    assert result["active_target_kind"] == "circle"


def test_release_report_accepts_active_paths(tmp_path) -> None:
    active_dir = tmp_path / "active_demo"
    active_dir.mkdir()
    (active_dir / "active_summary.json").write_text(
        json.dumps(
            {
                "kind": "ellipse",
                "strategy": "spread",
                "initial_reconstruction_score": 0.0,
                "final_reconstruction_score": 0.75,
                "score_delta": 0.75,
                "rounds": [{"round": 1, "trace_shape": [1, 10, 2]}],
            }
        ),
        encoding="utf-8",
    )
    out = generate_release_html_report(tmp_path / "report.html", active_paths=[active_dir])
    text = out.read_text(encoding="utf-8")
    assert "Artifact validation" in text
    assert "active" in text


def test_public_world_has_top_level_blind_markers() -> None:
    secret = make_default_world("ellipse")
    public = public_world_from_secret(secret, challenge="ellipse-easy")
    assert public["blind"] is True
    assert public["answer_hidden"] is True
    assert public["blind_schema_version"] == "0.9.1"
    assert public["blind_public_metadata"]["blind"] is True
    assert is_blind_public_world(public)


def test_blind_npz_world_json_has_top_level_markers(tmp_path) -> None:
    secret = make_default_world("ellipse")
    public = public_world_from_secret(secret, challenge="ellipse-easy")
    path = tmp_path / "public_obs.npz"
    save_run_npz(
        path,
        receiver_traces=np.zeros((1, 4, 2), dtype=np.float32),
        time=np.arange(4, dtype=np.float32),
        velocity_model=background_velocity_model_from_world(secret),
        source_coordinates=np.zeros((1, 2), dtype=np.float32),
        receiver_coordinates=np.zeros((2, 2), dtype=np.float32),
        final_wavefield=None,
        snapshots=None,
        world_json=json.dumps(public),
    )
    with np.load(path, allow_pickle=False) as data:
        world = json.loads(str(data["world_json"].item()))
    assert world["blind"] is True
    assert world["answer_hidden"] is True
    assert world["medium"]["anomaly"]["center_x"] == 0.5
