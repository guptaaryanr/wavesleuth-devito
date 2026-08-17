#!/usr/bin/env python3
"""Run the local pre-v1 release gate from a WaveSleuth-Devito checkout."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def static_checks(root: Path) -> None:
    active_path = root / "src" / "wavesleuth_devito" / "active.py"
    release_path = root / "src" / "wavesleuth_devito" / "release.py"
    active_source = active_path.read_text(encoding="utf-8")
    release_source = release_path.read_text(encoding="utf-8")

    tree = ast.parse(active_source)
    loaded = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "__version__"
        for node in ast.walk(tree)
    )
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "metadata"
        and any(alias.name == "__version__" for alias in node.names)
        for node in tree.body
    )
    assigned = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in tree.body
    )
    if loaded and not (imported or assigned):
        raise SystemExit("active.py loads __version__ without defining or importing it")

    stale = re.compile(
        r"\\bv[0-9]+\\.[0-9]+(?:\\.[0-9]+)? is a hardening(?: cleanup)? release\\b",
        flags=re.IGNORECASE,
    )
    if stale.search(release_source):
        raise SystemExit("release.py still contains a hard-coded historical hardening sentence")


def import_smoke(root: Path, env: dict[str, str]) -> None:
    code = r"""
import re
import tempfile
from pathlib import Path
import wavesleuth_devito.active as active
from wavesleuth_devito import __version__
from wavesleuth_devito.metadata import ARTIFACT_SCHEMA_VERSION
from wavesleuth_devito.release import RELEASE_REPORT_NOTE, generate_release_html_report
assert active.__version__ == __version__
assert active.ARTIFACT_SCHEMA_VERSION == ARTIFACT_SCHEMA_VERSION
if hasattr(active, "_ws_v092_original_run_active_demo"):
    original = active._ws_v092_original_run_active_demo
    try:
        active._ws_v092_original_run_active_demo = lambda *args, **kwargs: {"rounds": []}
        result = active.run_active_demo()
        assert result["version"] == __version__
        assert result["schema_version"] == ARTIFACT_SCHEMA_VERSION
    finally:
        active._ws_v092_original_run_active_demo = original
with tempfile.TemporaryDirectory() as tmp:
    path = generate_release_html_report(Path(tmp) / "report.html")
    text = path.read_text(encoding="utf-8")
    assert f"WaveSleuth-Devito v{__version__} Release Report" in text
    assert RELEASE_REPORT_NOTE in text
    assert not re.search(r"\\bv[0-9]+\\.[0-9]+(?:\\.[0-9]+)? is a hardening(?: cleanup)? release\\b", text, re.I)
"""
    run([sys.executable, "-c", code], cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-devito", action="store_true", help="Also run Devito doctor and self-test checks.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip the full pytest suite.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")

    static_checks(root)
    import_smoke(root, env)
    run([sys.executable, "-m", "compileall", "-q", "src", "tests"], cwd=root, env=env)
    if not args.skip_pytest:
        run([sys.executable, "-m", "pytest"], cwd=root, env=env)
    run([sys.executable, "-m", "pip", "check"], cwd=root, env=env)
    run(["wavesleuth-devito", "doctor"], cwd=root, env=env)
    run(["wavesleuth-devito", "self-test"], cwd=root, env=env)
    if args.with_devito:
        run(["wavesleuth-devito", "doctor", "--try-devito"], cwd=root, env=env)
        run(["wavesleuth-devito", "self-test", "--try-devito"], cwd=root, env=env)
    print("Pre-v1 gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
