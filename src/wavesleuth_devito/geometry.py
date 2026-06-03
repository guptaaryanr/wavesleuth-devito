"""Geometry utilities for physical coordinates and grid coordinates."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .exceptions import ValidationError


def grid_shape(world: dict[str, Any]) -> tuple[int, int]:
    """Return `(nx, nz)` from a world dictionary."""
    grid = world["grid"]
    return int(grid["nx"]), int(grid["nz"])


def grid_extent(world: dict[str, Any]) -> tuple[float, float]:
    """Return `(extent_x, extent_z)` from a world dictionary."""
    grid = world["grid"]
    return float(grid["extent_x"]), float(grid["extent_z"])


def grid_spacing(world: dict[str, Any]) -> tuple[float, float]:
    """Return physical grid spacing `(dx, dz)`."""
    nx, nz = grid_shape(world)
    extent_x, extent_z = grid_extent(world)
    if nx < 2 or nz < 2:
        raise ValidationError("Grid dimensions must both be at least 2.")
    return extent_x / float(nx - 1), extent_z / float(nz - 1)


def coordinate_vectors(world: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D physical coordinate arrays for x and z."""
    nx, nz = grid_shape(world)
    extent_x, extent_z = grid_extent(world)
    return (
        np.linspace(0.0, extent_x, nx, dtype=np.float32),
        np.linspace(0.0, extent_z, nz, dtype=np.float32),
    )


def coordinate_mesh(world: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D coordinate meshes with shape `(nx, nz)`."""
    x, z = coordinate_vectors(world)
    return np.meshgrid(x, z, indexing="ij")


def points_to_array(points: Iterable[dict[str, Any]], *, label: str = "points") -> np.ndarray:
    """Convert dictionaries with `x` and `z` fields into an `(n, 2)` float array."""
    rows: list[tuple[float, float]] = []
    for idx, point in enumerate(points):
        try:
            x = float(point["x"])
            z = float(point["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid {label}[{idx}]; expected numeric x and z fields.") from exc
        rows.append((x, z))
    if not rows:
        raise ValidationError(f"At least one {label} entry is required.")
    return np.asarray(rows, dtype=np.float32)


def source_coordinates(world: dict[str, Any]) -> np.ndarray:
    """Return source coordinates as an `(n_sources, 2)` array."""
    return points_to_array(world["acquisition"].get("sources", []), label="sources")


def receiver_coordinates(world: dict[str, Any]) -> np.ndarray:
    """Return receiver coordinates as an `(n_receivers, 2)` array."""
    return points_to_array(world["acquisition"].get("receivers", []), label="receivers")


def check_points_inside_domain(world: dict[str, Any], points: np.ndarray, *, label: str) -> None:
    """Validate that every point lies inside the physical domain."""
    extent_x, extent_z = grid_extent(world)
    for idx, (x, z) in enumerate(points):
        if not (0.0 <= float(x) <= extent_x and 0.0 <= float(z) <= extent_z):
            raise ValidationError(
                f"{label}[{idx}] = ({float(x):.6g}, {float(z):.6g}) lies outside "
                f"domain [0, {extent_x}] x [0, {extent_z}]."
            )


def physical_to_grid_index(world: dict[str, Any], x: float, z: float) -> tuple[int, int]:
    """Map physical coordinates to the nearest grid indices."""
    nx, nz = grid_shape(world)
    extent_x, extent_z = grid_extent(world)
    if not (0.0 <= x <= extent_x and 0.0 <= z <= extent_z):
        raise ValidationError(f"Point ({x}, {z}) lies outside the world domain.")
    ix = int(round((x / extent_x) * (nx - 1))) if extent_x > 0 else 0
    iz = int(round((z / extent_z) * (nz - 1))) if extent_z > 0 else 0
    return max(0, min(nx - 1, ix)), max(0, min(nz - 1, iz))


def candidate_axes(
    world: dict[str, Any],
    grid_size: int,
    *,
    margin: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return x and z axes for a square candidate search grid."""
    if grid_size < 2:
        raise ValidationError("candidate grid size must be at least 2.")
    extent_x, extent_z = grid_extent(world)
    if margin is None:
        margin = 0.12 * min(extent_x, extent_z)
    margin = float(margin)
    max_margin = 0.45 * min(extent_x, extent_z)
    if margin < 0.0:
        raise ValidationError("candidate margin must be non-negative.")
    if margin >= max_margin:
        margin = max_margin
    xs = np.linspace(margin, extent_x - margin, grid_size, dtype=np.float32)
    zs = np.linspace(margin, extent_z - margin, grid_size, dtype=np.float32)
    return xs, zs, margin


def candidate_centers(
    world: dict[str, Any],
    grid_size: int,
    *,
    margin: float | None = None,
) -> list[dict[str, float | int]]:
    """Return candidate center dictionaries in row-major z/x order."""
    xs, zs, used_margin = candidate_axes(world, grid_size, margin=margin)
    centers: list[dict[str, float | int]] = []
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            centers.append(
                {
                    "index_x": ix,
                    "index_z": iz,
                    "center_x": float(x),
                    "center_z": float(z),
                    "margin": float(used_margin),
                }
            )
    return centers
