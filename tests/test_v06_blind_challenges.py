from __future__ import annotations

import json

import numpy as np

from wavesleuth_devito.blind import blind_observed_run, challenge_secret_digest, is_blind_public_world, public_world_from_secret
from wavesleuth_devito.challenge import score_challenge_directory
from wavesleuth_devito.io import load_json, load_run_npz, save_json, world_from_run
from wavesleuth_devito.scoring import score_reconstruction
from wavesleuth_devito.world import background_velocity_model_from_world, make_default_world, velocity_model_from_world


def test_public_world_redacts_circle_center() -> None:
    secret = make_default_world("circle")
    public = public_world_from_secret(secret, challenge="circle-easy")
    assert is_blind_public_world(public)
    assert public["medium"]["anomaly"]["radius"] == secret["medium"]["anomaly"]["radius"]
    assert public["medium"]["anomaly"]["center_x"] != secret["medium"]["anomaly"]["center_x"]
    assert public["medium"]["anomaly"]["center_z"] != secret["medium"]["anomaly"]["center_z"]
    assert challenge_secret_digest(secret) == challenge_secret_digest(json.loads(json.dumps(secret)))


def test_blind_observed_run_redacts_model_and_world(tmp_path) -> None:
    secret = make_default_world("circle")
    public = public_world_from_secret(secret, challenge="circle-easy")
    secret_run = tmp_path / "secret_run.npz"
    public_run = tmp_path / "public_run.npz"
    np.savez_compressed(
        secret_run,
        receiver_traces=np.ones((4, 2), dtype=np.float32),
        time=np.arange(4, dtype=np.float32),
        velocity_model=velocity_model_from_world(secret),
        source_coordinates=np.zeros((1, 2), dtype=np.float32),
        receiver_coordinates=np.zeros((2, 2), dtype=np.float32),
        final_wavefield=np.ones((3, 3), dtype=np.float32),
        snapshots=np.ones((2, 3, 3), dtype=np.float32),
        world_json=np.asarray(json.dumps(secret)),
        shot_mode=np.asarray("sequential"),
    )
    blind_observed_run(secret_run, public_run, public)
    run = load_run_npz(public_run)
    loaded = world_from_run(run)
    assert is_blind_public_world(loaded)
    assert np.allclose(run["velocity_model"], background_velocity_model_from_world(public))
    assert not np.allclose(run["velocity_model"], velocity_model_from_world(secret))
    assert run["final_wavefield"].size == 0
    assert run["snapshots"].size == 0


def test_score_challenge_directory_uses_secret_world(tmp_path) -> None:
    secret = make_default_world("circle")
    anomaly = secret["medium"]["anomaly"]
    secret_path = tmp_path / "secret" / "circle-easy_secret_world.json"
    recon_path = tmp_path / "runs" / "circle-easy_recon.json"
    secret_path.parent.mkdir(parents=True)
    recon_path.parent.mkdir(parents=True)
    save_json(secret, secret_path)
    reconstruction = {
        "best_candidate": {
            "center_x": anomaly["center_x"],
            "center_z": anomaly["center_z"],
            "radius": anomaly["radius"],
            "anomaly_velocity": secret["medium"]["anomaly_velocity"],
        }
    }
    save_json(reconstruction, recon_path)
    save_json({"challenge": "circle-easy", "blind": True, "secret_world_path": str(secret_path), "reconstruction_path": str(recon_path)}, tmp_path / "challenge_summary.json")
    result = score_challenge_directory(tmp_path, update_reconstruction=True)
    assert result["physical_score"]["supported"] is True
    assert result["physical_score"]["center_error"] == 0.0
    updated = load_json(recon_path)
    assert updated["physical_score"]["iou"] == score_reconstruction(secret, reconstruction)["iou"]
