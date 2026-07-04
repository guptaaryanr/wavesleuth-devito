from __future__ import annotations

import subprocess
import sys

from wavesleuth_devito.io import load_world


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wavesleuth_devito.cli", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cli_help() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "generate-world" in result.stdout


def test_cli_generate_world(tmp_path) -> None:
    out = tmp_path / "circle.json"
    result = run_cli("generate-world", "--kind", "circle", "--out", str(out))
    assert result.returncode == 0, result.stderr
    loaded = load_world(out)
    assert loaded["medium"]["anomaly"]["kind"] == "circle"
