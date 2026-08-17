from __future__ import annotations
import re
from pathlib import Path
from wavesleuth_devito import ARTIFACT_SCHEMA_VERSION, __version__
from wavesleuth_devito.release import RELEASE_REPORT_NOTE, RELEASE_SCHEMA_VERSION, doctor_report, generate_release_html_report, version_tuple
from wavesleuth_devito.world import make_default_world


def test_v1_metadata_contract() -> None:
    assert version_tuple(__version__) >= (1, 0, 0)
    assert ARTIFACT_SCHEMA_VERSION == __version__
    assert RELEASE_SCHEMA_VERSION == __version__
    assert make_default_world("circle")["schema_version"] == __version__
    assert doctor_report()["package"]["version"] == __version__


def test_pyproject_and_runtime_versions_match() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match and match.group(1) == __version__
    assert "Development Status :: 5 - Production/Stable" in text


def test_release_report_is_v1_and_version_dynamic(tmp_path) -> None:
    path = generate_release_html_report(tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert f"WaveSleuth-Devito v{__version__} Release Report" in text
    assert RELEASE_REPORT_NOTE in text
    assert "release candidate" not in text.lower()


def test_v1_docs_and_release_tools_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "README.md", "docs/SCHEMA.md", "docs/FEATURE_MATRIX.md",
        "docs/V1_RELEASE_NOTES.md", "docs/POST_V1_ROADMAP.md",
        "docs/V1_RELEASE_CHECKLIST.md", "examples/v1_release.md",
        "tools/release_gate.py", "tools/pre_v1_gate.py",
    ):
        assert (root / relative).exists(), relative
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    assert "v1.0.0 is the first stable release" in readme
    assert "release candidate" not in readme
    gate = (root / "tools" / "release_gate.py").read_text(encoding="utf-8")
    assert "wheel_smoke" in gate and "pip" in gate and "wheel" in gate
    wrapper = (root / "tools" / "pre_v1_gate.py").read_text(encoding="utf-8")
    assert "from release_gate import main" in wrapper
