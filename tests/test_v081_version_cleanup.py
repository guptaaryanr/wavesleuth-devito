from __future__ import annotations

from pathlib import Path

from wavesleuth_devito.metadata import __version__


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    return tuple(int(part) for part in parts[:3])  # type: ignore[return-value]


def test_version_is_at_least_v081() -> None:
    assert _version_tuple(__version__) >= (0, 8, 1)


def test_legacy_version_regression_tests_use_minimum_checks() -> None:
    tests_dir = Path(__file__).parent
    legacy_files = [
        tests_dir / "test_v072_active_shape_compat.py",
        tests_dir / "test_v08_mask_blocks.py",
    ]
    forbidden = [
        'assert __version__ == ' + '"0.7.2"',
        'assert __version__ == ' + '"0.8.0"',
    ]
    for path in legacy_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text
