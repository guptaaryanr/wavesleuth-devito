from __future__ import annotations

import inspect
import re
from pathlib import Path

from wavesleuth_devito import __version__
from wavesleuth_devito.active import run_active_demo
from wavesleuth_devito.blind import PUBLIC_SCHEMA_VERSION, public_world_from_secret
from wavesleuth_devito.challenge import CHALLENGE_METADATA
from wavesleuth_devito.metadata import ARTIFACT_SCHEMA_VERSION
from wavesleuth_devito.release import RELEASE_SCHEMA_VERSION, generate_release_html_report
from wavesleuth_devito.world import make_default_world


def _version_tuple(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split(".")[:3])  # type: ignore[return-value]


def test_version_and_schema_are_at_least_v093() -> None:
    assert _version_tuple(__version__) >= (0, 9, 3)
    assert ARTIFACT_SCHEMA_VERSION == __version__
    assert RELEASE_SCHEMA_VERSION == ARTIFACT_SCHEMA_VERSION
    assert PUBLIC_SCHEMA_VERSION == ARTIFACT_SCHEMA_VERSION


def test_blind_public_world_uses_current_schema() -> None:
    public = public_world_from_secret(make_default_world("ellipse"), challenge="ellipse-easy")
    assert public["blind_schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_generated_world_uses_current_schema() -> None:
    assert make_default_world("circle")["schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_active_summary_source_uses_dynamic_schema() -> None:
    source = inspect.getsource(run_active_demo)
    assert "schema_version" in source
    assert not re.search(
        r'["\']schema_version["\']\s*(?::|=)\s*["\']0\.[0-9]+\.[0-9]+["\']',
        source,
    )


def test_challenge_descriptions_are_version_neutral() -> None:
    parts: list[str] = []
    for item in CHALLENGE_METADATA.values():
        parts.append(str(item.get("description", "")))
        parts.extend(str(note) for note in item.get("notes", []))
    assert not re.search(r"\bv0\.[0-9]", " ".join(parts))


def test_release_report_is_dynamic_and_version_neutral(tmp_path) -> None:
    out = generate_release_html_report(tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    assert f"WaveSleuth-Devito v{__version__} Release Report" in text
    assert "v0.9.1 is a hardening" not in text


def test_public_docs_and_hygiene_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "README.md",
        "docs/SCHEMA.md",
        "docs/FEATURE_MATRIX.md",
        "docs/ROADMAP_TO_V1.md",
        "docs/POST_V1_ROADMAP.md",
        "docs/V1_RELEASE_CHECKLIST.md",
        "examples/quickstart.md",
        "examples/v1_release_candidate.md",
        "tools/release_hygiene.py",
    ):
        assert (root / relative).exists(), relative
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "## Current capabilities" in readme
    assert "## v0.3 improvements" not in readme
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "wavesleuth_v1_audit_*/" in gitignore


def test_no_stale_current_version_assertions() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r'assert\s+__version__\s*==\s*"0\.[0-9]+\.[0-9]+"')
    offenders = []
    for path in (root / "tests").glob("test*.py"):
        if path.name == Path(__file__).name:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []
