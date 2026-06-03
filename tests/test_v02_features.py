from __future__ import annotations

import numpy as np

from wavesleuth_devito.scoring import normalize_trace_channels, trace_mismatch, window_trace_data
from wavesleuth_devito.world import background_velocity_model_from_world, make_default_world


def test_crossfire_world_has_multiple_sources() -> None:
    world = make_default_world("circle", acquisition="crossfire")
    assert len(world["acquisition"]["sources"]) == 3
    assert world["simulation"]["shot_mode"] == "sequential"


def test_background_model_is_homogeneous_for_circle() -> None:
    world = make_default_world("circle")
    bg = background_velocity_model_from_world(world)
    assert bg.shape == (world["grid"]["nx"], world["grid"]["nz"])
    assert np.allclose(bg, world["medium"]["background_velocity"])


def test_trace_mismatch_supports_3d_trace_cube() -> None:
    observed = np.ones((2, 5, 3), dtype=np.float32)
    simulated = observed.copy()
    assert trace_mismatch(observed, simulated) == 0.0


def test_trace_time_window() -> None:
    data = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    time = np.linspace(0.0, 1.0, 5)
    windowed = window_trace_data(data, time, time_min=0.25, time_max=0.75)
    assert windowed.shape == (2, 3, 3)


def test_trace_channel_normalization() -> None:
    data = np.ones((2, 5, 3), dtype=np.float32)
    normalized = normalize_trace_channels(data)
    norms = np.sqrt(np.sum(normalized * normalized, axis=1))
    assert np.allclose(norms, 1.0)


def test_correlation_mismatch_scale_invariant() -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    simulated = 5.0 * observed
    assert np.isclose(trace_mismatch(observed, simulated, metric="correlation"), 0.0)
