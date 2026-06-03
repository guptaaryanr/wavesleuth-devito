from __future__ import annotations

import numpy as np
import pytest

from wavesleuth_devito.inversion import select_best_candidate
from wavesleuth_devito.scoring import trace_mismatch
from wavesleuth_devito.simulation import simulate_world
from wavesleuth_devito.world import make_demo_world


def test_select_best_candidate() -> None:
    records = [
        {"center_x": 0.2, "center_z": 0.2, "mismatch": 3.0},
        {"center_x": 0.5, "center_z": 0.5, "mismatch": 1.0},
        {"center_x": 0.8, "center_z": 0.8, "mismatch": 2.0},
    ]
    best = select_best_candidate(records)
    assert best["center_x"] == 0.5
    assert best["mismatch"] == 1.0


def test_trace_mismatch_zero_for_identical_traces() -> None:
    traces = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert trace_mismatch(traces, traces) == 0.0


@pytest.mark.devito
def test_tiny_devito_forward_if_available() -> None:
    pytest.importorskip("devito")
    world = make_demo_world()
    world["grid"].update({"nx": 20, "nz": 20, "extent_x": 0.30, "extent_z": 0.30})
    world["medium"]["anomaly"].update({"center_x": 0.15, "center_z": 0.15, "radius": 0.035})
    world["acquisition"]["sources"] = [{"x": 0.08, "z": 0.08}]
    world["acquisition"]["receivers"] = [{"x": 0.14, "z": 0.20}, {"x": 0.20, "z": 0.20}]
    world["simulation"].update({"nt": 30, "dt": 0.0008, "source_frequency": 25.0, "space_order": 2})
    result = simulate_world(world, save_wavefield=False, quiet=True)
    assert result.receiver_traces.shape == (30, 2)
