#!/usr/bin/env python3
"""Run the WaveSleuth-Devito release gate."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

MIN_RELEASE = (1, 0, 0)


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")[:3]
    nums = [int("".join(ch for ch in part if ch.isdigit()) or 0) for part in parts]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def current_version(root: Path) -> str:
    text = (root / "src" / "wavesleuth_devito" / "metadata.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise SystemExit("Could not determine package version")
    return match.group(1)


def project_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise SystemExit("Could not determine pyproject version")
    return match.group(1)


def static_checks(root: Path, expected: str) -> None:
    if version_tuple(expected) < MIN_RELEASE:
        raise SystemExit(f"Release gate requires v1.0.0 or newer, found {expected}")
    if project_version(root) != expected:
        raise SystemExit("Runtime and pyproject versions differ")

    active_source = (root / "src" / "wavesleuth_devito" / "active.py").read_text(encoding="utf-8")
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
        raise SystemExit("active.py loads __version__ without defining/importing it")

    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    if "release candidate" in readme or "final pre-v1" in readme:
        raise SystemExit("README still presents v1 as pending")


def import_smoke(root: Path, env: dict[str, str], expected: str) -> None:
    code = f"""
import tempfile
from pathlib import Path
from wavesleuth_devito import __version__
from wavesleuth_devito.metadata import ARTIFACT_SCHEMA_VERSION
from wavesleuth_devito.release import RELEASE_REPORT_NOTE, RELEASE_SCHEMA_VERSION, generate_release_html_report
from wavesleuth_devito.world import make_default_world
assert __version__ == {expected!r}
assert ARTIFACT_SCHEMA_VERSION == __version__
assert RELEASE_SCHEMA_VERSION == __version__
assert make_default_world('circle')['schema_version'] == __version__
with tempfile.TemporaryDirectory() as tmp:
    path = generate_release_html_report(Path(tmp) / 'report.html')
    text = path.read_text(encoding='utf-8')
    assert f'WaveSleuth-Devito v{{__version__}} Release Report' in text
    assert RELEASE_REPORT_NOTE in text
    assert 'release candidate' not in text.lower()
"""
    run([sys.executable, "-c", code], cwd=root, env=env)


def _venv_exe(root: Path, name: str) -> Path:
    if os.name == "nt":
        suffix = ".exe" if name in {"python", "wavesleuth-devito"} else ""
        return root / "Scripts" / f"{name}{suffix}"
    return root / "bin" / name


def inspect_wheel(wheel: Path, expected: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        entry_names = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_names) != 1:
            raise SystemExit("Wheel metadata or entry points are missing")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        entries = archive.read(entry_names[0]).decode("utf-8")
        if "Name: wavesleuth-devito" not in metadata or f"Version: {expected}" not in metadata:
            raise SystemExit("Wheel project metadata is incorrect")
        for dep in ("numpy", "matplotlib"):
            if not re.search(rf"^Requires-Dist: {dep}(?:[ ;<>=]|$)", metadata, re.MULTILINE | re.IGNORECASE):
                raise SystemExit(f"Wheel is missing dependency metadata for {dep}")
        if "Requires-Python: >=3.10" not in metadata:
            raise SystemExit("Wheel Python requirement is incorrect")
        if "wavesleuth-devito = wavesleuth_devito.cli:main" not in entries:
            raise SystemExit("Wheel console entry point is missing")
        if "wavesleuth_devito/metadata.py" not in names:
            raise SystemExit("Wheel package modules are missing")


def wheel_smoke(root: Path, env: dict[str, str], expected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="wavesleuth-wheel-") as tmp_raw:
        tmp = Path(tmp_raw)
        wheelhouse = tmp / "wheelhouse"
        wheelhouse.mkdir()
        run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",  "--wheel-dir", str(wheelhouse), "."],
            cwd=root,
            env=env,
        )
        wheels = sorted(wheelhouse.glob("wavesleuth_devito-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"Expected one wheel, found {wheels}")
        wheel = wheels[0]
        inspect_wheel(wheel, expected)

        venv_dir = tmp / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=True).create(venv_dir)
        py = _venv_exe(venv_dir, "python")
        cli = _venv_exe(venv_dir, "wavesleuth-devito")
        clean_env = dict(env)
        clean_env.pop("PYTHONPATH", None)
        run([str(py), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=tmp, env=clean_env)
        code = f"""
from importlib.metadata import version
from wavesleuth_devito import __version__
from wavesleuth_devito.metadata import ARTIFACT_SCHEMA_VERSION
assert version('wavesleuth-devito') == {expected!r}
assert __version__ == {expected!r}
assert ARTIFACT_SCHEMA_VERSION == __version__
"""
        run([str(py), "-c", code], cwd=tmp, env=clean_env)
        run([str(cli), "--version"], cwd=tmp, env=clean_env)
        run([str(cli), "doctor"], cwd=tmp, env=clean_env)
        run([str(cli), "self-test"], cwd=tmp, env=clean_env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the WaveSleuth-Devito release gate")
    parser.add_argument("--with-devito", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-pip-check", action="store_true")
    parser.add_argument("--skip-wheel", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected = current_version(root)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")

    static_checks(root, expected)
    import_smoke(root, env, expected)
    run([sys.executable, "-m", "compileall", "-q", "src", "tests"], cwd=root, env=env)
    if not args.skip_pytest:
        run([sys.executable, "-m", "pytest"], cwd=root, env=env)
    if not args.skip_pip_check:
        run([sys.executable, "-m", "pip", "check"], cwd=root, env=env)
    cli = [sys.executable, "-m", "wavesleuth_devito.cli"]
    run([*cli, "doctor"], cwd=root, env=env)
    run([*cli, "self-test"], cwd=root, env=env)
    if args.with_devito:
        run([*cli, "doctor", "--try-devito"], cwd=root, env=env)
        run([*cli, "self-test", "--try-devito"], cwd=root, env=env)
    if not args.skip_wheel:
        wheel_smoke(root, env, expected)
    print(f"WaveSleuth-Devito v{expected} release gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
