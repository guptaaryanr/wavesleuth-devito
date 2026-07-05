"""Blind challenge helpers for separating public observations from secret answers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ValidationError
from .io import array_string, ensure_parent
from .world import background_velocity_model_from_world, validate_world

PUBLIC_SCHEMA_VERSION = "0.6.1"


def canonical_json(data: dict[str, Any]) -> str:
    """Return deterministic compact JSON for challenge hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the exact bytes stored on disk."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def challenge_secret_canonical_digest(secret_world: dict[str, Any]) -> str:
    """Return a SHA-256 digest of canonical compact secret-world JSON.

    This digest is stable under formatting/key-order changes. It is useful for
    comparing two JSON objects semantically, but it is not necessarily the same
    value produced by shell tools such as ``sha256sum secret_world.json``.
    """
    validate_world(secret_world)
    return hashlib.sha256(canonical_json(secret_world).encode("utf-8")).hexdigest()


def challenge_secret_digest(secret_world: dict[str, Any]) -> str:
    """Backward-compatible alias for the canonical secret-world digest."""
    return challenge_secret_canonical_digest(secret_world)


def secret_world_hashes(secret_world: dict[str, Any], file_path: str | Path | None = None) -> dict[str, str | None]:
    """Return explicit canonical and file-byte SHA-256 digests for a secret world.

    ``secret_world_sha256`` is the file-byte digest when a file exists because
    that is what users can verify directly with ``sha256sum``. The canonical
    digest is also recorded for object-level comparisons.
    """
    canonical = challenge_secret_canonical_digest(secret_world)
    file_digest = sha256_file(file_path) if file_path is not None and Path(file_path).exists() else None
    return {
        "secret_world_sha256": file_digest or canonical,
        "secret_world_file_sha256": file_digest,
        "secret_world_canonical_sha256": canonical,
    }


def is_blind_public_world(world: dict[str, Any]) -> bool:
    """Return True if `world` is public metadata for a blind challenge."""
    if not isinstance(world, dict):
        return False
    marker = world.get("blind_public_metadata")
    if isinstance(marker, dict):
        return bool(marker.get("blind", marker.get("answer_hidden", False)))
    if bool(marker):
        return True
    marker = world.get("challenge")
    return isinstance(marker, dict) and bool(marker.get("blind_public_metadata", False))


def public_world_from_secret(secret_world: dict[str, Any], *, challenge: str | None = None) -> dict[str, Any]:
    """Return valid public world metadata that hides the true anomaly location.

    The current baseline inversions still need target family and known-shape
    hints such as circle radius or ellipse axes. v0.6 hides the answer-bearing
    location and model arrays; fully secret shape/contrast challenges are left
    for later releases.
    """
    validate_world(secret_world)
    public = copy.deepcopy(secret_world)
    grid = public["grid"]
    mid_x = 0.5 * float(grid["extent_x"])
    mid_z = 0.5 * float(grid["extent_z"])
    public["name"] = f"{secret_world.get('name', challenge or 'challenge')}_public"
    anomaly = public.get("medium", {}).get("anomaly", {})
    kind = str(anomaly.get("kind"))

    if kind in {"circle", "ellipse", "rectangle", "ring", "crack", "circle-layered"}:
        if "center_x" in anomaly:
            anomaly["center_x"] = mid_x
        if "center_z" in anomaly:
            anomaly["center_z"] = mid_z
    elif kind == "mask-blocks":
        anomaly["active_cells"] = [{"i": 0, "j": 0}]
    elif kind == "two-circles":
        circles = anomaly.get("circles", [])
        offsets = [-0.08 * float(grid["extent_x"]), 0.08 * float(grid["extent_x"])]
        for idx, circle in enumerate(circles):
            circle["center_x"] = mid_x + offsets[min(idx, len(offsets) - 1)]
            circle["center_z"] = mid_z
    elif kind == "blobs":
        for blob in anomaly.get("blobs", []):
            blob["center_x"] = mid_x
            blob["center_z"] = mid_z
    elif kind == "layered":
        pass
    else:
        raise ValidationError(f"Cannot build blind public metadata for anomaly kind {kind!r}.")

    public["blind_public_metadata"] = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "blind": True,
        "answer_hidden": True,
        "challenge": challenge,
        "redacted_fields": [
            "hidden anomaly location",
            "true velocity_model array",
            "final_wavefield",
            "snapshots",
        ],
        "known_to_inversion": [
            "grid",
            "acquisition geometry",
            "simulation parameters",
            "background velocity",
            "target family",
            "known-shape hints used by the baseline solver",
        ],
        "notes": [
            "This public world is a metadata carrier for blind inversion; its anomaly center is a placeholder, not the answer.",
            "Use a secret answer world or score-challenge for final scoring.",
        ],
    }
    validate_world(public)
    return public


def blind_observed_run(secret_run_path: str | Path, public_run_path: str | Path, public_world: dict[str, Any]) -> Path:
    """Write a public observed run that preserves traces but redacts answer arrays."""
    validate_world(public_world)
    src = Path(secret_run_path)
    out = ensure_parent(public_run_path)
    if not src.exists():
        raise ValidationError(f"Secret run file not found: {src}")
    with np.load(src, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    required = {"receiver_traces", "time", "source_coordinates", "receiver_coordinates", "world_json"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValidationError(f"Secret run is missing arrays required for blind export: {', '.join(missing)}")
    payload["world_json"] = np.asarray(json.dumps(public_world, sort_keys=True))
    payload["velocity_model"] = background_velocity_model_from_world(public_world).astype(np.float32)
    payload["final_wavefield"] = np.empty((0,), dtype=np.float32)
    payload["snapshots"] = np.empty((0,), dtype=np.float32)
    payload["blind_public_run"] = np.asarray("true")
    payload["blind_schema_version"] = np.asarray(PUBLIC_SCHEMA_VERSION)
    np.savez_compressed(out, **payload)
    return out


def run_is_blind_public(run: dict[str, np.ndarray]) -> bool:
    """Return True if a loaded run is marked as a blind public observation."""
    raw = run.get("blind_public_run")
    if raw is not None and array_string(raw).lower() == "true":
        return True
    raw_world = run.get("world_json")
    if raw_world is None:
        return False
    try:
        world = json.loads(array_string(raw_world))
    except Exception:
        return False
    return is_blind_public_world(world)
