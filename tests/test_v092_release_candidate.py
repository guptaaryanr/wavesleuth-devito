from __future__ import annotations

from wavesleuth_devito.metadata import __version__
from wavesleuth_devito.release import generate_release_html_report, version_tuple


def test_v092_release_candidate_version() -> None:
    assert version_tuple(__version__) >= (0, 9, 2)


def test_release_report_title_tracks_package_version(tmp_path) -> None:
    out = generate_release_html_report(tmp_path / "release_report.html")
    text = out.read_text(encoding="utf-8")
    assert f"WaveSleuth-Devito v{__version__} Release Report" in text
