from __future__ import annotations

import json

import numpy as np

from wavesleuth_devito.io import load_json, load_run_npz, load_world, save_json, save_run_npz, save_world, world_from_run
from wavesleuth_devito.world import make_default_world


def test_json_save_load(tmp_path) -> None:
    path = tmp_path / "data.json"
    save_json({"b": 2, "a": 1}, path)
    assert load_json(path) == {"a": 1, "b": 2}


def test_world_save_load(tmp_path) -> None:
    world = make_default_world("circle")
    path = tmp_path / "world.json"
    save_world(world, path)
    loaded = load_world(path)
    assert loaded["name"] == world["name"]


def test_run_npz_roundtrip(tmp_path) -> None:
    world = make_default_world("circle")
    path = tmp_path / "run.npz"
    save_run_npz(
        path,
        receiver_traces=np.zeros((4, 2), dtype=np.float32),
        time=np.arange(4, dtype=np.float32),
        velocity_model=np.ones((3, 3), dtype=np.float32),
        source_coordinates=np.zeros((1, 2), dtype=np.float32),
        receiver_coordinates=np.zeros((2, 2), dtype=np.float32),
        final_wavefield=None,
        snapshots=None,
        world_json=json.dumps(world),
    )
    run = load_run_npz(path)
    assert run["receiver_traces"].shape == (4, 2)
    assert world_from_run(run)["name"] == world["name"]
