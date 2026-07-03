from __future__ import annotations

import inspect

from wavesleuth_devito.challenge import SUPPORTED_CHALLENGES, make_challenge_world
from wavesleuth_devito.inversion import SUPPORTED_SEARCH_STRATEGIES, _parameter_penalty, grid_search_circle


def test_staged_strategy_is_supported() -> None:
    assert "staged" in SUPPORTED_SEARCH_STRATEGIES
    signature = inspect.signature(grid_search_circle)
    assert "search_strategy" in signature.parameters
    assert "top_k_refine" in signature.parameters


def test_reference_parameter_prior_zero_at_reference() -> None:
    penalty = _parameter_penalty(
        radius=0.12,
        velocity=2.2,
        reference_radius=0.12,
        reference_velocity=2.2,
        parameter_prior="reference",
        radius_prior_weight=0.1,
        velocity_prior_weight=0.1,
    )
    assert penalty == 0.0


def test_reference_parameter_prior_positive_away_from_reference() -> None:
    penalty = _parameter_penalty(
        radius=0.09,
        velocity=1.99,
        reference_radius=0.12,
        reference_velocity=2.2,
        parameter_prior="reference",
        radius_prior_weight=0.1,
        velocity_prior_weight=0.1,
    )
    assert penalty > 0.0


def test_staged_radius_velocity_challenge_metadata() -> None:
    assert "circle-radius-velocity-staged" in SUPPORTED_CHALLENGES
    _world, settings = make_challenge_world("circle-radius-velocity-staged")
    assert settings["search_strategy"] == "staged"
    assert settings["search_radius"] is True
    assert settings["search_velocity"] is True
    assert settings["top_k_refine"] >= 1


def test_staged_grid_search_flow_with_mock_forward(monkeypatch) -> None:
    from types import SimpleNamespace

    import numpy as np

    import wavesleuth_devito.inversion as inv
    from wavesleuth_devito.world import make_demo_world

    world = make_demo_world()
    true = world["medium"]["anomaly"]
    true_velocity = float(world["medium"]["anomaly_velocity"])

    def signature(model: dict[str, float]) -> np.ndarray:
        if model.get("kind") == "background":
            values = np.zeros(4, dtype=np.float32)
        else:
            values = np.array(
                [
                    float(model["center_x"]),
                    float(model["center_z"]),
                    float(model["radius"]),
                    float(model["anomaly_velocity"]),
                ],
                dtype=np.float32,
            )
        return np.tile(values, (12, 1)).astype(np.float32)

    observed = signature(
        {
            "kind": "circle",
            "center_x": float(true["center_x"]),
            "center_z": float(true["center_z"]),
            "radius": float(true["radius"]),
            "anomaly_velocity": true_velocity,
        }
    )
    run = {"receiver_traces": observed, "time": np.arange(12, dtype=np.float32)}

    class FakeEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, model):
            return SimpleNamespace(receiver_traces=signature(model))

    monkeypatch.setattr(inv, "load_run_npz", lambda _path: run)
    monkeypatch.setattr(inv, "world_from_run", lambda _run: world)
    monkeypatch.setattr(inv, "ForwardTraceEngine", FakeEngine)
    monkeypatch.setattr(inv, "background_velocity_model_from_world", lambda _world: {"kind": "background"})

    def fake_velocity_model(candidate_world):
        anomaly = candidate_world["medium"]["anomaly"]
        return {
            "kind": "circle",
            "center_x": float(anomaly["center_x"]),
            "center_z": float(anomaly["center_z"]),
            "radius": float(anomaly["radius"]),
            "anomaly_velocity": float(candidate_world["medium"]["anomaly_velocity"]),
        }

    monkeypatch.setattr(inv, "velocity_model_from_world", fake_velocity_model)

    reconstruction = inv.grid_search_circle(
        "fake_run.npz",
        candidate_grid_size=5,
        search_radius=True,
        search_velocity=True,
        refine_levels=1,
        search_strategy="staged",
        quiet=True,
    )
    best = reconstruction["best_candidate"]
    assert reconstruction["search"]["search_strategy"] == "staged"
    assert reconstruction["score"]["center_error"] < 0.05
    assert abs(float(best["radius"]) - float(true["radius"])) < 1.0e-6
    assert abs(float(best["anomaly_velocity"]) - true_velocity) < 1.0e-6
