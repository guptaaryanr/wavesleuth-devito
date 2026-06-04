from __future__ import annotations

import json

import numpy as np

from wavesleuth_devito.challenge import make_challenge_world
from wavesleuth_devito.inversion import default_radius_values, default_velocity_values
from wavesleuth_devito.io import save_json
from wavesleuth_devito.report import generate_html_report
from wavesleuth_devito.scoring import budgeted_challenge_score, probability_map_from_mismatch_map
from wavesleuth_devito.simulation import apply_trace_noise, sponge_damping_model
from wavesleuth_devito.uncertainty import candidate_probabilities
from wavesleuth_devito.world import make_default_world


def test_ring_acquisition_and_sponge_profile() -> None:
    world = make_default_world("circle", acquisition="ring")
    assert len(world["acquisition"]["sources"]) == 4
    assert len(world["acquisition"]["receivers"]) >= 12
    world["simulation"].update({"boundary": "sponge", "sponge_width": 5, "sponge_strength": 12.0})
    damp = sponge_damping_model(world)
    assert damp.shape == (world["grid"]["nx"], world["grid"]["nz"])
    assert float(damp.max()) > 0.0
    assert float(damp[world["grid"]["nx"] // 2, world["grid"]["nz"] // 2]) == 0.0


def test_default_radius_and_velocity_values() -> None:
    assert default_radius_values(0.12) == [0.09, 0.12, 0.15]
    vals = default_velocity_values(1.5, 2.2)
    assert len(vals) == 3
    assert vals[1] == 2.2


def test_trace_noise_is_deterministic() -> None:
    traces = np.ones((2, 8, 3), dtype=np.float32)
    a = apply_trace_noise(traces, dt=0.001, noise_level=0.05, amplitude_jitter=0.02, seed=123)
    b = apply_trace_noise(traces, dt=0.001, noise_level=0.05, amplitude_jitter=0.02, seed=123)
    assert a.shape == traces.shape
    assert np.allclose(a, b)
    assert not np.allclose(a, traces)


def test_candidate_probabilities_sum_to_one() -> None:
    reconstruction = {
        "candidates": [
            {"center_x": 0.4, "center_z": 0.5, "radius": 0.1, "anomaly_velocity": 2.0, "mismatch": 2.0},
            {"center_x": 0.6, "center_z": 0.5, "radius": 0.1, "anomaly_velocity": 2.0, "mismatch": 1.0},
        ]
    }
    summary = candidate_probabilities(reconstruction, temperature=1.0)
    total = sum(float(c["probability"]) for c in summary["top_candidates"])
    assert np.isclose(total, 1.0)
    assert summary["top_candidates"][0]["center_x"] == 0.6


def test_probability_map_from_mismatch_map() -> None:
    probability, summary = probability_map_from_mismatch_map([[2.0, 1.0], [3.0, None]], temperature=1.0)
    assert probability.shape == (2, 2)
    assert np.isclose(float(probability.sum()), 1.0)
    assert summary["max_probability"] > 0.0


def test_budgeted_challenge_score_prefers_better_iou() -> None:
    good = budgeted_challenge_score({"supported": True, "iou": 0.8, "normalized_center_error": 0.02}, n_forward_runs=50, n_sources=3, n_receivers=8)
    bad = budgeted_challenge_score({"supported": True, "iou": 0.2, "normalized_center_error": 0.2}, n_forward_runs=50, n_sources=3, n_receivers=8)
    assert good["score"] > bad["score"]


def test_challenge_world_has_noise_config() -> None:
    world, settings = make_challenge_world("circle-noisy")
    assert settings["noise_level"] > 0.0
    assert world["medium"]["anomaly"]["kind"] == "circle"


def test_report_generation(tmp_path) -> None:
    world = make_default_world("circle")
    anomaly = world["medium"]["anomaly"]
    recon = {
        "method": "grid-search",
        "world_name": world["name"],
        "world": world,
        "objective": {"mismatch_mode": "differential", "metric": "l2"},
        "true_center": anomaly,
        "best_candidate": {
            "center_x": anomaly["center_x"],
            "center_z": anomaly["center_z"],
            "radius": anomaly["radius"],
            "mismatch": 0.0,
        },
        "score": {"iou": 1.0, "center_error": 0.0},
        "candidates": [{"center_x": anomaly["center_x"], "center_z": anomaly["center_z"], "mismatch": 0.0}],
    }
    recon_path = tmp_path / "recon.json"
    out_path = tmp_path / "report.html"
    save_json(recon, recon_path)
    out = generate_html_report(recon_path, out_path)
    assert out.exists()
    assert "WaveSleuth" in out.read_text(encoding="utf-8")
