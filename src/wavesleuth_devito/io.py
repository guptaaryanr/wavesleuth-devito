"""JSON and NPZ file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ValidationError


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for `path` and return a `Path`."""
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dictionary as stable, pretty JSON."""
    p = ensure_parent(path)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"File not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"Expected a JSON object in {p}.")
    return data


def save_world(world: dict[str, Any], path: str | Path) -> Path:
    """Validate and save a world JSON file."""
    from .world import validate_world

    validate_world(world)
    return save_json(world, path)


def load_world(path: str | Path) -> dict[str, Any]:
    """Load and validate a world JSON file."""
    from .world import validate_world

    world = load_json(path)
    validate_world(world)
    return world


def save_run_npz(
    path: str | Path,
    *,
    receiver_traces: np.ndarray,
    time: np.ndarray,
    velocity_model: np.ndarray,
    source_coordinates: np.ndarray,
    receiver_coordinates: np.ndarray,
    final_wavefield: np.ndarray | None,
    snapshots: np.ndarray | None,
    world_json: str,
    shot_mode: str | None = None,
) -> Path:
    """Save a simulation run as a compressed `.npz` file."""
    p = ensure_parent(path)
    final = np.asarray(final_wavefield if final_wavefield is not None else np.empty((0,), dtype=np.float32), dtype=np.float32)
    snaps = np.asarray(snapshots if snapshots is not None else np.empty((0,), dtype=np.float32), dtype=np.float32)
    payload: dict[str, np.ndarray] = {
        "receiver_traces": np.asarray(receiver_traces, dtype=np.float32),
        "time": np.asarray(time, dtype=np.float32),
        "velocity_model": np.asarray(velocity_model, dtype=np.float32),
        "source_coordinates": np.asarray(source_coordinates, dtype=np.float32),
        "receiver_coordinates": np.asarray(receiver_coordinates, dtype=np.float32),
        "final_wavefield": final,
        "snapshots": snaps,
        "world_json": np.asarray(world_json),
    }
    if shot_mode is not None:
        payload["shot_mode"] = np.asarray(str(shot_mode))
    np.savez_compressed(p, **payload)
    return p


def load_run_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a WaveSleuth `.npz` run into memory."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"Run file not found: {p}")
    try:
        with np.load(p, allow_pickle=False) as data:
            required = {
                "receiver_traces",
                "time",
                "velocity_model",
                "source_coordinates",
                "receiver_coordinates",
                "final_wavefield",
                "snapshots",
                "world_json",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValidationError(f"Run file {p} is missing arrays: {', '.join(missing)}")
            return {key: data[key].copy() for key in data.files}
    except OSError as exc:
        raise ValidationError(f"Could not load run file {p}: {exc}") from exc


def array_string(value: np.ndarray | str | bytes, *, default: str = "") -> str:
    """Extract a string saved as a scalar NPZ array."""
    try:
        if isinstance(value, np.ndarray):
            return str(value.item())
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    except Exception:
        return default


def world_from_run(run: dict[str, np.ndarray]) -> dict[str, Any]:
    """Extract and validate the world metadata stored inside a run dictionary."""
    raw = run.get("world_json")
    if raw is None:
        raise ValidationError("Run data is missing world_json metadata.")
    try:
        world_json = array_string(raw)
        world = json.loads(world_json)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValidationError("Run world_json metadata is invalid.") from exc
    from .world import validate_world

    validate_world(world)
    return world
