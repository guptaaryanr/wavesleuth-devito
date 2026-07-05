from __future__ import annotations

import json

from wavesleuth_devito.metadata import __version__
from wavesleuth_devito.release import (
    DEFAULT_RELEASE_CHALLENGES,
    doctor_report,
    generate_release_html_report,
    normalize_challenge_summary_schema,
    validate_artifact_path,
    validate_artifact_paths,
    version_tuple,
)
from wavesleuth_devito.io import save_json


def test_version_is_at_least_v09() -> None:
    assert version_tuple(__version__) >= (0, 9, 0)


def test_release_challenge_list_covers_current_milestones() -> None:
    assert "circle-easy" in DEFAULT_RELEASE_CHALLENGES
    assert "ellipse-easy" in DEFAULT_RELEASE_CHALLENGES
    assert "circle-radius-velocity-staged" in DEFAULT_RELEASE_CHALLENGES
    assert "mask-cell-easy" in DEFAULT_RELEASE_CHALLENGES


def test_normalize_challenge_summary_schema_preserves_aliases() -> None:
    summary = {
        "challenge": "circle-easy",
        "score": {"supported": True, "iou": 0.8},
        "challenge_score": {"supported": True, "score": 71.9},
    }
    normalized = normalize_challenge_summary_schema(summary)
    assert normalized["physical_score"]["iou"] == 0.8
    assert normalized["challenge_score"]["score"] == 71.9
    assert "schema_notes" in normalized


def test_doctor_report_is_lightweight() -> None:
    report = doctor_report()
    assert report["package"]["version"] == __version__
    assert report["modules"]["numpy"] is True
    assert report["modules"]["matplotlib"] is True


def test_validate_missing_challenge_directory_reports_error(tmp_path) -> None:
    report = validate_artifact_path(tmp_path / "missing_challenge")
    assert report["ok"] is False
    assert report["errors"]


def test_validate_artifact_paths_combines_results(tmp_path) -> None:
    unknown = tmp_path / "unknown.txt"
    unknown.write_text("not a wavesleuth artifact", encoding="utf-8")
    report = validate_artifact_paths([unknown])
    assert report["ok"] is False
    assert report["checked"] == 1


def test_release_html_report_writes(tmp_path) -> None:
    out = generate_release_html_report(tmp_path / "release_report.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "WaveSleuth-Devito v0.9 Release Report" in text
