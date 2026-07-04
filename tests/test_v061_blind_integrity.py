from __future__ import annotations

from wavesleuth_devito.blind import challenge_secret_canonical_digest, secret_world_hashes, sha256_file
from wavesleuth_devito.challenge import private_answer_reconstruction
from wavesleuth_devito.io import save_json
from wavesleuth_devito.visualization import visualize_reconstruction
from wavesleuth_devito.world import make_default_world


def test_secret_world_hashes_distinguish_file_and_canonical(tmp_path) -> None:
    world = make_default_world("circle")
    path = tmp_path / "secret_world.json"
    save_json(world, path)
    hashes = secret_world_hashes(world, path)
    assert hashes["secret_world_sha256"] == sha256_file(path)
    assert hashes["secret_world_file_sha256"] == sha256_file(path)
    assert hashes["secret_world_canonical_sha256"] == challenge_secret_canonical_digest(world)
    assert hashes["secret_world_file_sha256"] != hashes["secret_world_canonical_sha256"]


def test_private_answer_reconstruction_unhides_secret_world() -> None:
    world = make_default_world("ellipse")
    public_recon = {
        "answer_hidden": True,
        "blind": True,
        "world": {**world, "blind_public_metadata": {"blind": True, "answer_hidden": True}},
        "best_candidate": {
            "kind": "ellipse",
            "center_x": 0.54,
            "center_z": 0.50,
            "radius_x": 0.17,
            "radius_z": 0.095,
            "angle_degrees": 25.0,
            "mismatch": 0.1,
        },
    }
    answer = private_answer_reconstruction(public_recon, world)
    assert answer["answer_hidden"] is False
    assert answer["blind"] is False
    assert answer["world"] == world
    assert answer["true_center"]["center_x"] == world["medium"]["anomaly"]["center_x"]
    assert answer["private_answer_view"] is True


def test_private_answer_reconstruction_figure_is_writable(tmp_path) -> None:
    world = make_default_world("ellipse")
    anomaly = world["medium"]["anomaly"]
    recon = private_answer_reconstruction(
        {
            "answer_hidden": True,
            "world": world,
            "best_candidate": {
                "kind": "ellipse",
                "center_x": anomaly["center_x"],
                "center_z": anomaly["center_z"],
                "radius_x": anomaly["radius_x"],
                "radius_z": anomaly["radius_z"],
                "angle_degrees": anomaly["angle_degrees"],
                "mismatch": 0.0,
            },
            "objective": {"mismatch_mode": "differential", "metric": "l2"},
        },
        world,
    )
    out = visualize_reconstruction(recon, tmp_path / "answer.png")
    assert out.exists()
    assert out.stat().st_size > 0
