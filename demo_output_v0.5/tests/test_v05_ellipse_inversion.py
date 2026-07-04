from __future__ import annotations

import json

import numpy as np

from wavesleuth_devito.inversion import grid_search_ellipse
from wavesleuth_devito.io import save_run_npz
from wavesleuth_devito.world import make_default_world, velocity_model_from_world


class _FakeResult:
    def __init__(self, traces: np.ndarray) -> None:
        self.receiver_traces = traces.astype(np.float32)


class _FakeEngine:
    def __init__(self, world, **_kwargs) -> None:
        self.world = world
        nx = int(world["grid"]["nx"])
        nz = int(world["grid"]["nz"])
        self.x = np.linspace(0.0, float(world["grid"]["extent_x"]), nx, dtype=np.float32)
        self.z = np.linspace(0.0, float(world["grid"]["extent_z"]), nz, dtype=np.float32)
        self.xmesh, self.zmesh = np.meshgrid(self.x, self.z, indexing="ij")

    def run(self, velocity_model: np.ndarray) -> _FakeResult:
        background = float(self.world["medium"]["background_velocity"])
        delta = np.asarray(velocity_model, dtype=np.float64) - background
        mass = float(delta.sum())
        if abs(mass) < 1.0e-12:
            traces = np.zeros((1, 3), dtype=np.float32)
        else:
            traces = np.asarray(
                [[mass, float((delta * self.xmesh).sum() / mass), float((delta * self.zmesh).sum() / mass)]],
                dtype=np.float32,
            )
        return _FakeResult(traces)


def test_grid_search_ellipse_selects_true_center_with_mock_engine(tmp_path, monkeypatch) -> None:
    import wavesleuth_devito.inversion as inversion

    world = make_default_world("ellipse")
    world["grid"].update({"nx": 32, "nz": 32})
    world["medium"]["anomaly"].update({"center_x": 0.5, "center_z": 0.5, "angle_degrees": 0.0})
    true_model = velocity_model_from_world(world)
    observed = _FakeEngine(world).run(true_model).receiver_traces
    run_path = tmp_path / "ellipse_obs.npz"
    save_run_npz(
        run_path,
        receiver_traces=observed,
        time=np.asarray([0.0], dtype=np.float32),
        velocity_model=true_model,
        source_coordinates=np.asarray([[0.2, 0.2]], dtype=np.float32),
        receiver_coordinates=np.asarray([[0.8, 0.8]], dtype=np.float32),
        final_wavefield=None,
        snapshots=None,
        world_json=json.dumps(world),
    )
    monkeypatch.setattr(inversion, "ForwardTraceEngine", _FakeEngine)
    reconstruction = grid_search_ellipse(run_path, candidate_grid_size=5, refine_levels=0, quiet=True)
    best = reconstruction["best_candidate"]
    assert best["kind"] == "ellipse"
    assert abs(best["center_x"] - 0.5) < 1.0e-6
    assert abs(best["center_z"] - 0.5) < 1.0e-6
    assert reconstruction["score"]["iou"] > 0.99
