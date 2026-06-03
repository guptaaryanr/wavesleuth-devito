"""World generation, validation, and velocity-model construction."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import (
    check_points_inside_domain,
    coordinate_mesh,
    grid_extent,
    grid_shape,
    receiver_coordinates,
    source_coordinates,
)

DEFAULT_SEED = 20260203
SUPPORTED_WORLD_KINDS = ("circle", "rectangle", "layered", "blobs")
SUPPORTED_ACQUISITION_PRESETS = ("single", "crossfire")


def _single_acquisition() -> dict[str, list[dict[str, float]]]:
    """One-source, top-receiver acquisition used by the original MVP."""
    return {
        "sources": [{"x": 0.20, "z": 0.12}],
        "receivers": [
            {"x": 0.15, "z": 0.82},
            {"x": 0.30, "z": 0.82},
            {"x": 0.45, "z": 0.82},
            {"x": 0.60, "z": 0.82},
            {"x": 0.75, "z": 0.82},
            {"x": 0.90, "z": 0.82},
        ],
    }


def _crossfire_acquisition() -> dict[str, list[dict[str, float]]]:
    """Sparse multi-angle acquisition for a less ambiguous toy inverse problem.

    The sources are intended to be fired sequentially. Keeping receivers on more
    than one side of the domain makes the coarse search surface less degenerate
    than the original single-shot, top-receiver setup.
    """
    return {
        "sources": [
            {"x": 0.18, "z": 0.18},
            {"x": 0.82, "z": 0.18},
            {"x": 0.18, "z": 0.72},
        ],
        "receivers": [
            {"x": 0.20, "z": 0.84},
            {"x": 0.40, "z": 0.84},
            {"x": 0.60, "z": 0.84},
            {"x": 0.80, "z": 0.84},
            {"x": 0.88, "z": 0.30},
            {"x": 0.88, "z": 0.50},
            {"x": 0.88, "z": 0.70},
            {"x": 0.12, "z": 0.34},
            {"x": 0.12, "z": 0.54},
            {"x": 0.12, "z": 0.74},
        ],
    }


def acquisition_preset(name: str) -> dict[str, list[dict[str, float]]]:
    """Return a named acquisition preset."""
    if name == "single":
        return _single_acquisition()
    if name == "crossfire":
        return _crossfire_acquisition()
    raise UnsupportedWorldError(
        f"Unsupported acquisition preset {name!r}. Supported: {', '.join(SUPPORTED_ACQUISITION_PRESETS)}"
    )


def _base_world(name: str, kind: str, *, acquisition: str = "single") -> dict[str, Any]:
    return {
        "name": name,
        "grid": {
            "nx": 70,
            "nz": 70,
            "extent_x": 1.0,
            "extent_z": 1.0,
        },
        "medium": {
            "background_velocity": 1.5,
            "anomaly_velocity": 2.2,
            "anomaly": {"kind": kind},
        },
        "acquisition": acquisition_preset(acquisition),
        "simulation": {
            "nt": 360,
            "dt": 0.0015,
            "space_order": 4,
            "source_frequency": 20.0,
            "shot_mode": "sequential" if acquisition == "crossfire" else "simultaneous",
        },
    }


def make_default_world(
    kind: str = "circle",
    *,
    seed: int = DEFAULT_SEED,
    name: str | None = None,
    acquisition: str = "single",
) -> dict[str, Any]:
    """Create a deterministic world dictionary for a supported kind."""
    if kind not in SUPPORTED_WORLD_KINDS:
        raise UnsupportedWorldError(f"Unsupported world kind {kind!r}. Supported: {', '.join(SUPPORTED_WORLD_KINDS)}")

    world = _base_world(name or f"{kind}_demo", kind, acquisition=acquisition)
    anomaly = world["medium"]["anomaly"]

    if kind == "circle":
        anomaly.update({"center_x": 0.55, "center_z": 0.52, "radius": 0.12})
    elif kind == "rectangle":
        anomaly.update({"center_x": 0.56, "center_z": 0.53, "width": 0.24, "height": 0.16})
    elif kind == "layered":
        world["medium"].pop("anomaly_velocity", None)
        anomaly.update(
            {
                "layers": [
                    {"z_min": 0.00, "z_max": 0.34, "velocity": 1.35},
                    {"z_min": 0.34, "z_max": 0.68, "velocity": 1.70},
                    {"z_min": 0.68, "z_max": 1.00, "velocity": 2.05},
                ]
            }
        )
    elif kind == "blobs":
        rng = random.Random(seed)
        blobs: list[dict[str, float]] = []
        for _ in range(4):
            blobs.append(
                {
                    "center_x": round(rng.uniform(0.25, 0.78), 4),
                    "center_z": round(rng.uniform(0.28, 0.68), 4),
                    "radius": round(rng.uniform(0.055, 0.105), 4),
                    "velocity": round(rng.uniform(1.9, 2.35), 4),
                }
            )
        anomaly.update({"seed": int(seed), "blobs": blobs})

    validate_world(world)
    return world


def validate_world(world: dict[str, Any]) -> None:
    """Validate a world dictionary and raise `ValidationError` if malformed."""
    if not isinstance(world, dict):
        raise ValidationError("World must be a dictionary.")
    for key in ("name", "grid", "medium", "acquisition", "simulation"):
        if key not in world:
            raise ValidationError(f"World is missing required key {key!r}.")

    grid = world["grid"]
    for key in ("nx", "nz", "extent_x", "extent_z"):
        if key not in grid:
            raise ValidationError(f"World grid is missing {key!r}.")
    nx, nz = int(grid["nx"]), int(grid["nz"])
    extent_x, extent_z = float(grid["extent_x"]), float(grid["extent_z"])
    if nx < 5 or nz < 5:
        raise ValidationError("Grid nx and nz must both be at least 5 for this playground.")
    if extent_x <= 0.0 or extent_z <= 0.0:
        raise ValidationError("Grid extents must be positive.")

    medium = world["medium"]
    background_velocity = float(medium.get("background_velocity", 0.0))
    if background_velocity <= 0.0:
        raise ValidationError("medium.background_velocity must be positive.")
    anomaly = medium.get("anomaly")
    if not isinstance(anomaly, dict) or "kind" not in anomaly:
        raise ValidationError("medium.anomaly must be a dictionary with a kind field.")
    kind = anomaly["kind"]
    if kind not in SUPPORTED_WORLD_KINDS:
        raise UnsupportedWorldError(f"Unsupported anomaly kind {kind!r}.")

    if kind in {"circle", "rectangle", "blobs"}:
        anomaly_velocity = float(medium.get("anomaly_velocity", background_velocity))
        if anomaly_velocity <= 0.0:
            raise ValidationError("medium.anomaly_velocity must be positive.")

    if kind == "circle":
        for key in ("center_x", "center_z", "radius"):
            if key not in anomaly:
                raise ValidationError(f"Circle anomaly missing {key!r}.")
        if float(anomaly["radius"]) <= 0.0:
            raise ValidationError("Circle radius must be positive.")
    elif kind == "rectangle":
        for key in ("center_x", "center_z", "width", "height"):
            if key not in anomaly:
                raise ValidationError(f"Rectangle anomaly missing {key!r}.")
        if float(anomaly["width"]) <= 0.0 or float(anomaly["height"]) <= 0.0:
            raise ValidationError("Rectangle width and height must be positive.")
    elif kind == "layered":
        layers = anomaly.get("layers", [])
        if not isinstance(layers, list) or not layers:
            raise ValidationError("Layered world requires a non-empty layers list.")
        for idx, layer in enumerate(layers):
            for key in ("z_min", "z_max", "velocity"):
                if key not in layer:
                    raise ValidationError(f"Layer {idx} missing {key!r}.")
            if float(layer["velocity"]) <= 0.0:
                raise ValidationError(f"Layer {idx} velocity must be positive.")
    elif kind == "blobs":
        blobs = anomaly.get("blobs", [])
        if not isinstance(blobs, list) or not blobs:
            raise ValidationError("Blobs world requires a non-empty blobs list.")
        for idx, blob in enumerate(blobs):
            for key in ("center_x", "center_z", "radius"):
                if key not in blob:
                    raise ValidationError(f"Blob {idx} missing {key!r}.")
            if float(blob["radius"]) <= 0.0:
                raise ValidationError(f"Blob {idx} radius must be positive.")
            if "velocity" in blob and float(blob["velocity"]) <= 0.0:
                raise ValidationError(f"Blob {idx} velocity must be positive.")

    acquisition = world["acquisition"]
    if not isinstance(acquisition.get("sources"), list):
        raise ValidationError("acquisition.sources must be a list.")
    if not isinstance(acquisition.get("receivers"), list):
        raise ValidationError("acquisition.receivers must be a list.")
    src = source_coordinates(world)
    rec = receiver_coordinates(world)
    check_points_inside_domain(world, src, label="sources")
    check_points_inside_domain(world, rec, label="receivers")

    simulation = world["simulation"]
    for key in ("nt", "dt", "space_order", "source_frequency"):
        if key not in simulation:
            raise ValidationError(f"simulation missing {key!r}.")
    if int(simulation["nt"]) < 3:
        raise ValidationError("simulation.nt must be at least 3.")
    if float(simulation["dt"]) <= 0.0:
        raise ValidationError("simulation.dt must be positive.")
    if int(simulation["space_order"]) < 2:
        raise ValidationError("simulation.space_order must be at least 2.")
    if float(simulation["source_frequency"]) <= 0.0:
        raise ValidationError("simulation.source_frequency must be positive.")
    shot_mode = str(simulation.get("shot_mode", "simultaneous"))
    if shot_mode not in {"simultaneous", "sequential"}:
        raise ValidationError("simulation.shot_mode must be 'simultaneous' or 'sequential' when supplied.")


def anomaly_kind(world: dict[str, Any]) -> str:
    """Return the anomaly kind from a world."""
    return str(world["medium"]["anomaly"]["kind"])


def anomaly_mask_from_world(world: dict[str, Any], *, circle_override: dict[str, float] | None = None) -> np.ndarray:
    """Return a boolean anomaly mask with shape `(nx, nz)`."""
    validate_world(world)
    kind = anomaly_kind(world)
    anomaly = world["medium"]["anomaly"]
    xmesh, zmesh = coordinate_mesh(world)

    if circle_override is not None:
        center_x = float(circle_override["center_x"])
        center_z = float(circle_override["center_z"])
        radius = float(circle_override["radius"])
        return ((xmesh - center_x) ** 2 + (zmesh - center_z) ** 2) <= radius**2

    if kind == "circle":
        center_x = float(anomaly["center_x"])
        center_z = float(anomaly["center_z"])
        radius = float(anomaly["radius"])
        return ((xmesh - center_x) ** 2 + (zmesh - center_z) ** 2) <= radius**2

    if kind == "rectangle":
        center_x = float(anomaly["center_x"])
        center_z = float(anomaly["center_z"])
        width = float(anomaly["width"])
        height = float(anomaly["height"])
        return (np.abs(xmesh - center_x) <= width / 2.0) & (np.abs(zmesh - center_z) <= height / 2.0)

    if kind == "layered":
        background = float(world["medium"]["background_velocity"])
        model = velocity_model_from_world(world)
        return np.abs(model - background) > 1.0e-6

    if kind == "blobs":
        mask = np.zeros(grid_shape(world), dtype=bool)
        for blob in anomaly["blobs"]:
            center_x = float(blob["center_x"])
            center_z = float(blob["center_z"])
            radius = float(blob["radius"])
            mask |= ((xmesh - center_x) ** 2 + (zmesh - center_z) ** 2) <= radius**2
        return mask

    raise UnsupportedWorldError(f"Unsupported anomaly kind {kind!r}.")


def velocity_model_from_world(world: dict[str, Any]) -> np.ndarray:
    """Create a 2D velocity model from a world dictionary."""
    validate_world(world)
    nx, nz = grid_shape(world)
    medium = world["medium"]
    background = float(medium["background_velocity"])
    model = np.full((nx, nz), background, dtype=np.float32)
    kind = anomaly_kind(world)
    anomaly = medium["anomaly"]

    if kind == "circle":
        model[anomaly_mask_from_world(world)] = float(medium["anomaly_velocity"])
    elif kind == "rectangle":
        model[anomaly_mask_from_world(world)] = float(medium["anomaly_velocity"])
    elif kind == "layered":
        _xmesh, zmesh = coordinate_mesh(world)
        extent_z = grid_extent(world)[1]
        for layer in anomaly["layers"]:
            z_min = float(layer["z_min"])
            z_max = float(layer["z_max"])
            if z_max <= 1.0 and extent_z != 1.0:
                z_min *= extent_z
                z_max *= extent_z
            layer_mask = (zmesh >= z_min) & (zmesh <= z_max)
            model[layer_mask] = float(layer["velocity"])
    elif kind == "blobs":
        xmesh, zmesh = coordinate_mesh(world)
        default_velocity = float(medium.get("anomaly_velocity", background))
        for blob in anomaly["blobs"]:
            center_x = float(blob["center_x"])
            center_z = float(blob["center_z"])
            radius = float(blob["radius"])
            velocity = float(blob.get("velocity", default_velocity))
            blob_mask = ((xmesh - center_x) ** 2 + (zmesh - center_z) ** 2) <= radius**2
            model[blob_mask] = velocity
    else:
        raise UnsupportedWorldError(f"Unsupported anomaly kind {kind!r}.")

    return model


def background_velocity_model_from_world(world: dict[str, Any]) -> np.ndarray:
    """Return the background model used by differential inversion.

    For circle/rectangle/blob worlds this is a homogeneous background. For a
    layered world, the generated velocity model is already a background-like
    structure, so it is returned as-is.
    """
    validate_world(world)
    kind = anomaly_kind(world)
    if kind == "layered":
        return velocity_model_from_world(world)
    nx, nz = grid_shape(world)
    background = float(world["medium"]["background_velocity"])
    return np.full((nx, nz), background, dtype=np.float32)


def circle_parameters(world: dict[str, Any]) -> dict[str, float] | None:
    """Return circle parameters if the world contains a circular anomaly."""
    if anomaly_kind(world) != "circle":
        return None
    anomaly = world["medium"]["anomaly"]
    return {
        "center_x": float(anomaly["center_x"]),
        "center_z": float(anomaly["center_z"]),
        "radius": float(anomaly["radius"]),
    }


def world_with_circle_candidate(
    world: dict[str, Any],
    *,
    center_x: float,
    center_z: float,
    radius: float,
    anomaly_velocity: float,
    name: str | None = None,
) -> dict[str, Any]:
    """Return a copy of `world` with a circular candidate anomaly."""
    candidate = copy.deepcopy(world)
    candidate["name"] = name or f"candidate_circle_{center_x:.3f}_{center_z:.3f}"
    candidate["medium"]["anomaly_velocity"] = float(anomaly_velocity)
    candidate["medium"]["anomaly"] = {
        "kind": "circle",
        "center_x": float(center_x),
        "center_z": float(center_z),
        "radius": float(radius),
    }
    validate_world(candidate)
    return candidate


def make_demo_world() -> dict[str, Any]:
    """Return a small crossfire circle world for the end-to-end demo."""
    world = make_default_world("circle", name="wavesleuth_demo", acquisition="crossfire")
    world["grid"].update({"nx": 52, "nz": 52, "extent_x": 1.0, "extent_z": 1.0})
    world["simulation"].update(
        {"nt": 360, "dt": 0.0015, "space_order": 4, "source_frequency": 18.0, "shot_mode": "sequential"}
    )
    validate_world(world)
    return world
