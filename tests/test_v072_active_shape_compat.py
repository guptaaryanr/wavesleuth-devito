from __future__ import annotations

import inspect

import numpy as np

from wavesleuth_devito.active import _run_trace_metadata, _standardize_active_run_trace_shape
from wavesleuth_devito.metadata import __version__
from wavesleuth_devito.simulation import ForwardTraceEngine


def test_version_is_v072() -> None:
    assert tuple(int(part) for part in __version__.split(".")[:3]) >= (0, 7, 2)


def test_forward_trace_engine_keeps_single_source_backward_compatible() -> None:
    source = inspect.getsource(ForwardTraceEngine.__init__)
    assert 'self.mode == "simultaneous" or self.src_coords.shape[0] == 1' in source
    assert "Sequential shot mode should have a stable" not in source


def test_standardize_active_run_trace_shape_promotes_2d_to_3d(tmp_path) -> None:
    path = tmp_path / "run.npz"
    np.savez_compressed(
        path,
        receiver_traces=np.zeros((5, 2), dtype=np.float32),
        time=np.arange(5, dtype=np.float32),
        world_json=np.asarray("{}"),
    )

    changed = _standardize_active_run_trace_shape(path)
    assert changed is True

    with np.load(path, allow_pickle=False) as data:
        assert data["receiver_traces"].shape == (1, 5, 2)
        assert data["time"].shape == (5,)
        assert str(data["world_json"].item()) == "{}"

    meta = _run_trace_metadata(path)
    assert meta["layout"] == "shot_time_receiver"
    assert meta["shape"] == [1, 5, 2]
    assert meta["active_trace_shape_standard"] is True


def test_standardize_active_run_trace_shape_keeps_3d(tmp_path) -> None:
    path = tmp_path / "run.npz"
    np.savez_compressed(path, receiver_traces=np.zeros((2, 5, 2), dtype=np.float32))

    changed = _standardize_active_run_trace_shape(path)
    assert changed is False

    with np.load(path, allow_pickle=False) as data:
        assert data["receiver_traces"].shape == (2, 5, 2)
