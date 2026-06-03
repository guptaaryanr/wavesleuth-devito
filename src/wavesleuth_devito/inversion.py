"""Simple search-based inversion routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import candidate_axes
from .io import load_run_npz, save_json, world_from_run
from .metadata import base_metadata
from .scoring import score_circle_reconstruction, trace_mismatch
from .simulation import DevitoAcoustic2D
from .world import anomaly_kind, circle_parameters, velocity_model_from_world, world_with_circle_candidate


def select_best_candidate(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the candidate with the smallest mismatch."""
    if not candidate_records:
        raise ValidationError("No candidate records were provided.")
    return min(candidate_records, key=lambda record: float(record["mismatch"]))


def grid_search_circle(
    run_path: str | Path,
    *,
    out_path: str | Path | None = None,
    candidate_grid_size: int = 5,
    radius: float | None = None,
    anomaly_velocity: float | None = None,
    max_candidates: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a coarse grid-search inversion for a circular anomaly center."""
    run = load_run_npz(run_path)
    world = world_from_run(run)
    if anomaly_kind(world) != "circle":
        raise UnsupportedWorldError("The MVP grid-search inversion currently supports circle worlds only.")

    observed = np.asarray(run["receiver_traces"], dtype=np.float32)
    true_params = circle_parameters(world)
    if true_params is None:
        raise UnsupportedWorldError("The observed run does not contain circle metadata.")

    used_radius = float(radius if radius is not None else true_params["radius"])
    used_velocity = float(
        anomaly_velocity
        if anomaly_velocity is not None
        else world["medium"].get("anomaly_velocity", world["medium"]["background_velocity"])
    )
    if used_radius <= 0.0:
        raise ValidationError("Candidate radius must be positive.")
    if used_velocity <= 0.0:
        raise ValidationError("Candidate anomaly velocity must be positive.")
    if max_candidates is not None and max_candidates < 1:
        raise ValidationError("max_candidates must be positive when supplied.")

    margin = max(used_radius * 1.10, 0.10 * min(float(world["grid"]["extent_x"]), float(world["grid"]["extent_z"])))
    xs, zs, used_margin = candidate_axes(world, candidate_grid_size, margin=margin)
    mismatch_map: list[list[float | None]] = [[None for _ in range(len(xs))] for _ in range(len(zs))]
    candidate_records: list[dict[str, Any]] = []

    solver = DevitoAcoustic2D(world, save_wavefield=False, quiet=quiet)
    total = len(xs) * len(zs)
    limit = total if max_candidates is None else min(total, int(max_candidates))
    count = 0

    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            if count >= limit:
                break
            count += 1
            candidate_world = world_with_circle_candidate(
                world,
                center_x=float(x),
                center_z=float(z),
                radius=used_radius,
                anomaly_velocity=used_velocity,
            )
            velocity_model = velocity_model_from_world(candidate_world)
            simulated = solver.run(velocity_model).receiver_traces
            mismatch = trace_mismatch(observed, simulated)
            mismatch_map[iz][ix] = float(mismatch)
            record = {
                "index_x": int(ix),
                "index_z": int(iz),
                "center_x": float(x),
                "center_z": float(z),
                "radius": float(used_radius),
                "anomaly_velocity": float(used_velocity),
                "mismatch": float(mismatch),
            }
            candidate_records.append(record)
            if not quiet:
                print(
                    f"candidate {count:03d}/{limit:03d}: "
                    f"center=({float(x):.3f}, {float(z):.3f}) mismatch={mismatch:.6g}"
                )
        if count >= limit:
            break

    best = select_best_candidate(candidate_records)
    score = score_circle_reconstruction(
        world,
        predicted_center_x=float(best["center_x"]),
        predicted_center_z=float(best["center_z"]),
        predicted_radius=float(best["radius"]),
        best_mismatch=float(best["mismatch"]),
    )

    reconstruction: dict[str, Any] = {
        **base_metadata(),
        "method": "grid-search",
        "run_path": str(run_path),
        "world_name": world.get("name", "unknown"),
        "world": world,
        "true_center": {
            "center_x": float(true_params["center_x"]),
            "center_z": float(true_params["center_z"]),
            "radius": float(true_params["radius"]),
        },
        "candidate_grid": {
            "grid_size": int(candidate_grid_size),
            "xs": [float(v) for v in xs],
            "zs": [float(v) for v in zs],
            "margin": float(used_margin),
            "evaluated_candidates": int(len(candidate_records)),
            "possible_candidates": int(total),
        },
        "mismatch_map": mismatch_map,
        "candidates": candidate_records,
        "best_candidate": best,
        "best_mismatch": float(best["mismatch"]),
        "score": score,
        "notes": [
            "MVP grid search assumes a circular anomaly.",
            "Radius and anomaly velocity are fixed from metadata unless CLI overrides are supplied.",
            "The forward model uses a simple Devito acoustic solver with crude zero-style boundaries, not a tuned PML.",
        ],
    }
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction
