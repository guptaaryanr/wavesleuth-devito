
from __future__ import annotations

from wavesleuth_devito.challenge import make_challenge_world
from wavesleuth_devito.metadata import __version__


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".")[:3])


def test_version_is_at_least_v082() -> None:
    assert _version_tuple(__version__) >= (0, 8, 2)


def test_mask_cell_easy_uses_calibrated_baseline_world() -> None:
    world, settings = make_challenge_world("mask-cell-easy")
    assert world["medium"]["anomaly"]["kind"] == "mask-blocks"
    assert world["grid"]["nx"] == 54
    assert world["grid"]["nz"] == 54
    assert world["simulation"]["shot_mode"] == "sequential"
    assert world["simulation"]["boundary"] == "none"
    assert int(world["simulation"].get("sponge_width", 0)) == 0
    assert float(world["simulation"].get("sponge_strength", 0.0)) == 0.0
    assert settings["method"] == "cell-search"
    assert int(settings["cell_grid_size"]) == 6
    assert int(settings["max_active_cells"]) == 5
