from __future__ import annotations

import inspect

from wavesleuth_devito.challenge import make_challenge_world
import wavesleuth_devito.challenge as challenge_module
from wavesleuth_devito.metadata import __version__


def _version_tuple(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split(".")[:3])  # type: ignore[return-value]


def test_version_is_at_least_v083() -> None:
    assert _version_tuple(__version__) >= (0, 8, 3)


def test_mask_cell_easy_uses_amplitude_sensitive_objective() -> None:
    world, settings = make_challenge_world("mask-cell-easy")
    assert world["grid"]["nx"] == 54
    assert world["grid"]["nz"] == 54
    assert world["simulation"]["boundary"] == "none"
    assert world["simulation"].get("sponge_width", 0) == 0
    assert settings["method"] == "cell-search"
    assert settings["mismatch_mode"] == "differential"
    assert settings["metric"] == "l2"
    assert settings.get("normalize_traces") is False


def test_mask_cell_challenge_passes_normalize_traces_to_cell_search() -> None:
    source = inspect.getsource(challenge_module)
    assert "normalize_traces=bool(settings.get(\"normalize_traces\", False))" in source
