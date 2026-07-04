from __future__ import annotations

from wavesleuth_devito.active import (
    active_source_pool,
    default_active_world,
    estimated_center_from_reconstruction,
    select_next_source,
)


def test_default_active_world_circle() -> None:
    world = default_active_world("circle")
    assert world["simulation"]["shot_mode"] == "sequential"
    assert world["simulation"]["boundary"] == "sponge"
    assert len(world["acquisition"]["sources"]) == 1
    assert len(world["acquisition"]["receivers"]) >= 8


def test_active_source_pool_unique() -> None:
    world = default_active_world("circle")
    pool = active_source_pool(world)
    assert len(pool) == len({(p["x"], p["z"]) for p in pool})
    assert len(pool) >= 8
    assert all(0.0 <= p["x"] <= 1.0 and 0.0 <= p["z"] <= 1.0 for p in pool)


def test_select_next_source_avoids_used() -> None:
    world = default_active_world("circle")
    used = [{"x": 0.18, "z": 0.18}]
    reconstruction = {
        "best_candidate": {"center_x": 0.55, "center_z": 0.52, "mismatch": 1.0},
        "candidates": [
            {"center_x": 0.55, "center_z": 0.52, "mismatch": 1.0},
            {"center_x": 0.50, "center_z": 0.50, "mismatch": 1.2},
        ],
    }
    selected = select_next_source(world, reconstruction, used, strategy="uncertainty")
    assert (selected["x"], selected["z"]) != (0.18, 0.18)
    assert selected["strategy"] == "uncertainty"
    assert "ranked_candidates" in selected


def test_estimated_center_from_best_candidate_fallback() -> None:
    estimate = estimated_center_from_reconstruction({"best_candidate": {"center_x": 0.4, "center_z": 0.6}})
    assert abs(estimate["center_x"] - 0.4) < 1.0e-12
    assert abs(estimate["center_z"] - 0.6) < 1.0e-12
