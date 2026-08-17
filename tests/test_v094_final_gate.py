from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import wavesleuth_devito.active as active
from wavesleuth_devito import __version__
from wavesleuth_devito.metadata import ARTIFACT_SCHEMA_VERSION
from wavesleuth_devito.release import (
    RELEASE_REPORT_NOTE,
    RELEASE_SCHEMA_VERSION,
    generate_release_html_report,
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split(".")[:3])  # type: ignore[return-value]


def _top_level_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_v094_versions_and_schema_are_consistent() -> None:
    assert _version_tuple(__version__) >= (0, 9, 4)
    assert ARTIFACT_SCHEMA_VERSION == __version__
    assert RELEASE_SCHEMA_VERSION == ARTIFACT_SCHEMA_VERSION
    assert active.__version__ == __version__
    assert active.ARTIFACT_SCHEMA_VERSION == ARTIFACT_SCHEMA_VERSION


def test_active_module_does_not_load_an_undefined_version_name() -> None:
    source = Path(active.__file__).read_text(encoding="utf-8")
    loaded_version = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "__version__"
        for node in ast.walk(ast.parse(source))
    )
    if loaded_version:
        assert "__version__" in _top_level_names(source)


def test_active_metadata_wrapper_executes_without_devito(monkeypatch) -> None:
    """Exercise the metadata wrapper that failed in the v0.9.3 audit.

    The original numerical function is replaced with a tiny dictionary, so this
    checks wrapper name resolution and artifact metadata without compiling a
    Devito operator.
    """
    original_name = "_ws_v092_original_run_active_demo"
    if not hasattr(active, original_name):
        # Repositories without the compatibility wrapper are covered by the
        # static symbol test and normal active-demo integration audit.
        return
    monkeypatch.setattr(active, original_name, lambda *args, **kwargs: {"rounds": []})
    result = active.run_active_demo()
    assert result["version"] == __version__
    assert result["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert result["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_release_report_has_only_dynamic_current_metadata(tmp_path) -> None:
    out = generate_release_html_report(tmp_path / "release_report.html")
    text = out.read_text(encoding="utf-8")
    assert f"WaveSleuth-Devito v{__version__} Release Report" in text
    assert RELEASE_REPORT_NOTE in text
    assert not re.search(
        r"\\bv[0-9]+\\.[0-9]+(?:\\.[0-9]+)? is a hardening(?: cleanup)? release\\b",
        text,
        flags=re.IGNORECASE,
    )


def test_runtime_sources_contain_no_stale_hardening_release_sentence() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r"\\bv[0-9]+\\.[0-9]+(?:\\.[0-9]+)? is a hardening(?: cleanup)? release\\b",
        flags=re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in sorted((root / "src" / "wavesleuth_devito").glob("*.py")):
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []


def test_pre_v1_gate_tool_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "pre_v1_gate.py").exists()
