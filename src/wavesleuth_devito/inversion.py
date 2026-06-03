"""Simple search-based inversion routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import candidate_axes, grid_extent
from .io import array_string, load_run_npz, save_json, world_from_run
from .metadata import base_metadata
from .scoring import center_error, score_circle_reconstruction, trace_mismatch
from .simulation import ForwardTraceEngine
from .world import (
    anomaly_kind,
    background_velocity_model_from_world,
    circle_parameters,
    velocity_model_from_world,
    world_with_circle_candidate,
)


def select_best_candidate(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the candidate with the smallest mismatch."""
    if not candidate_records:
        raise ValidationError("No candidate records were provided.")
    return min(candidate_records, key=lambda record: float(record["mismatch"]))


def _shot_mode_from_run(run: dict[str, np.ndarray], observed: np.ndarray) -> str:
    if "shot_mode" in run:
        mode = array_string(run["shot_mode"], default="")
        if mode in {"simultaneous", "sequential"}:
            return mode
    return "sequential" if observed.ndim == 3 else "simultaneous"


def _candidate_grid_around(
    world: dict[str, Any],
    *,
    grid_size: int,
    margin: float,
    center_x: float | None = None,
    center_z: float | None = None,
    span_x: float | None = None,
    span_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if center_x is None or center_z is None or span_x is None or span_z is None:
        return candidate_axes(world, grid_size, margin=margin)
    extent_x, extent_z = grid_extent(world)
    half_x = max(float(span_x) / 2.0, 1.0e-6)
    half_z = max(float(span_z) / 2.0, 1.0e-6)
    lo_x = max(float(margin), float(center_x) - half_x)
    hi_x = min(float(extent_x) - float(margin), float(center_x) + half_x)
    lo_z = max(float(margin), float(center_z) - half_z)
    hi_z = min(float(extent_z) - float(margin), float(center_z) + half_z)
    if hi_x <= lo_x:
        hi_x = lo_x + 1.0e-6
    if hi_z <= lo_z:
        hi_z = lo_z + 1.0e-6
    return (
        np.linspace(lo_x, hi_x, grid_size, dtype=np.float32),
        np.linspace(lo_z, hi_z, grid_size, dtype=np.float32),
        float(margin),
    )


def _nearest_true_summary(candidates: list[dict[str, Any]], true_params: dict[str, float]) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = sorted(candidates, key=lambda c: float(c["mismatch"]))
    nearest = min(
        candidates,
        key=lambda c: center_error(
            (true_params["center_x"], true_params["center_z"]),
            (float(c["center_x"]), float(c["center_z"])),
        ),
    )
    rank = 1 + next(i for i, c in enumerate(ranked) if c is nearest)
    return {
        "candidate": nearest,
        "rank_by_mismatch": int(rank),
        "distance_to_true_center": center_error(
            (true_params["center_x"], true_params["center_z"]),
            (float(nearest["center_x"]), float(nearest["center_z"])),
        ),
    }


def grid_search_circle(
    run_path: str | Path,
    *,
    out_path: str | Path | None = None,
    candidate_grid_size: int = 5,
    radius: float | None = None,
    anomaly_velocity: float | None = None,
    max_candidates: int | None = None,
    quiet: bool = False,
    mismatch_mode: str = "differential",
    metric: str = "l2",
    time_min: float | None = None,
    time_max: float | None = None,
    normalize_traces: bool = False,
    refine_levels: int = 0,
    shot_mode: str | None = None,
) -> dict[str, Any]:
    """Run a coarse grid-search inversion for a circular anomaly center.

    v0.2 adds a differential objective: compare anomaly-scattered residuals
    `(hidden traces - background traces)` instead of raw traces. This removes
    much of the direct-arrival dominance that made the original MVP happily pick
    wrong locations with nearly unchanged raw waveforms.
    """
    run = load_run_npz(run_path)
    world = world_from_run(run)
    if anomaly_kind(world) != "circle":
        raise UnsupportedWorldError("The MVP grid-search inversion currently supports circle worlds only.")

    observed = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
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
    if candidate_grid_size < 2:
        raise ValidationError("candidate_grid_size must be at least 2.")
    if max_candidates is not None and max_candidates < 1:
        raise ValidationError("max_candidates must be positive when supplied.")
    if refine_levels < 0:
        raise ValidationError("refine_levels must be non-negative.")
    mismatch_mode = mismatch_mode.lower()
    if mismatch_mode not in {"raw", "differential"}:
        raise ValidationError("mismatch_mode must be 'raw' or 'differential'.")

    used_shot_mode = shot_mode or _shot_mode_from_run(run, observed)
    if used_shot_mode not in {"simultaneous", "sequential"}:
        raise ValidationError("shot_mode must be 'simultaneous' or 'sequential'.")

    engine = ForwardTraceEngine(world, shot_mode=used_shot_mode, save_wavefield=False, quiet=quiet)

    margin = max(used_radius * 1.10, 0.10 * min(float(world["grid"]["extent_x"]), float(world["grid"]["extent_z"])))
    background_traces: np.ndarray | None = None
    if mismatch_mode == "differential":
        background_model = background_velocity_model_from_world(world)
        background_traces = engine.run(background_model).receiver_traces
        if background_traces.shape != observed.shape:
            raise ValidationError(
                f"Background traces shape {background_traces.shape} does not match observed shape {observed.shape}. "
                "Check shot_mode and source geometry."
            )
        observed_target = observed - background_traces
    else:
        observed_target = observed

    candidate_records: list[dict[str, Any]] = []
    search_levels: list[dict[str, Any]] = []
    total_possible_per_level = candidate_grid_size * candidate_grid_size
    evaluated_total = 0
    previous_best: dict[str, Any] | None = None
    previous_dx: float | None = None
    previous_dz: float | None = None
    cache: dict[tuple[int, int], dict[str, Any]] = {}

    for level in range(int(refine_levels) + 1):
        if level == 0 or previous_best is None or previous_dx is None or previous_dz is None:
            xs, zs, used_margin = _candidate_grid_around(world, grid_size=candidate_grid_size, margin=margin)
        else:
            xs, zs, used_margin = _candidate_grid_around(
                world,
                grid_size=candidate_grid_size,
                margin=margin,
                center_x=float(previous_best["center_x"]),
                center_z=float(previous_best["center_z"]),
                span_x=previous_dx,
                span_z=previous_dz,
            )
        dx = float(xs[1] - xs[0]) if len(xs) > 1 else previous_dx or 0.0
        dz = float(zs[1] - zs[0]) if len(zs) > 1 else previous_dz or 0.0
        mismatch_map: list[list[float | None]] = [[None for _ in range(len(xs))] for _ in range(len(zs))]
        level_records: list[dict[str, Any]] = []

        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                if max_candidates is not None and evaluated_total >= int(max_candidates):
                    break
                key = (round(float(x), 8), round(float(z), 8))
                cached = cache.get(key)
                if cached is not None:
                    mismatch_map[iz][ix] = float(cached["mismatch"])
                    level_records.append(cached)
                    continue
                candidate_world = world_with_circle_candidate(
                    world,
                    center_x=float(x),
                    center_z=float(z),
                    radius=used_radius,
                    anomaly_velocity=used_velocity,
                )
                velocity_model = velocity_model_from_world(candidate_world)
                simulated = engine.run(velocity_model).receiver_traces
                if simulated.shape != observed.shape:
                    raise ValidationError(f"Candidate traces shape {simulated.shape} does not match observed shape {observed.shape}.")
                simulated_target = simulated - background_traces if background_traces is not None else simulated
                mismatch = trace_mismatch(
                    observed_target,
                    simulated_target,
                    metric=metric,
                    time=time,
                    time_min=time_min,
                    time_max=time_max,
                    normalize_traces=normalize_traces,
                )
                evaluated_total += 1
                mismatch_map[iz][ix] = float(mismatch)
                record = {
                    "level": int(level),
                    "index_x": int(ix),
                    "index_z": int(iz),
                    "center_x": float(x),
                    "center_z": float(z),
                    "radius": float(used_radius),
                    "anomaly_velocity": float(used_velocity),
                    "mismatch": float(mismatch),
                }
                candidate_records.append(record)
                level_records.append(record)
                cache[key] = record
                if not quiet:
                    limit_text = "?" if max_candidates is None else str(max_candidates)
                    print(
                        f"level {level} candidate {evaluated_total}/{limit_text}: "
                        f"center=({float(x):.3f}, {float(z):.3f}) mismatch={mismatch:.6g}"
                    )
            if max_candidates is not None and evaluated_total >= int(max_candidates):
                break

        if level_records:
            previous_best = select_best_candidate(level_records)
            previous_dx = abs(dx) if dx else previous_dx
            previous_dz = abs(dz) if dz else previous_dz
        search_levels.append(
            {
                "level": int(level),
                "grid_size": int(candidate_grid_size),
                "xs": [float(v) for v in xs],
                "zs": [float(v) for v in zs],
                "margin": float(used_margin),
                "mismatch_map": mismatch_map,
                "best_candidate": previous_best,
                "evaluated_candidates_this_level": int(len(level_records)),
            }
        )
        if max_candidates is not None and evaluated_total >= int(max_candidates):
            break

    best = select_best_candidate(candidate_records)
    score = score_circle_reconstruction(
        world,
        predicted_center_x=float(best["center_x"]),
        predicted_center_z=float(best["center_z"]),
        predicted_radius=float(best["radius"]),
        best_mismatch=float(best["mismatch"]),
    )
    nearest = _nearest_true_summary(candidate_records, true_params)
    final_level = search_levels[-1]

    reconstruction: dict[str, Any] = {
        **base_metadata(),
        "method": "grid-search",
        "run_path": str(run_path),
        "world_name": world.get("name", "unknown"),
        "world": world,
        "objective": {
            "mismatch_mode": mismatch_mode,
            "metric": metric,
            "time_min": None if time_min is None else float(time_min),
            "time_max": None if time_max is None else float(time_max),
            "normalize_traces": bool(normalize_traces),
            "shot_mode": used_shot_mode,
            "background_subtracted": bool(mismatch_mode == "differential"),
        },
        "true_center": {
            "center_x": float(true_params["center_x"]),
            "center_z": float(true_params["center_z"]),
            "radius": float(true_params["radius"]),
        },
        "candidate_grid": {
            "grid_size": int(candidate_grid_size),
            "xs": final_level["xs"],
            "zs": final_level["zs"],
            "margin": float(final_level["margin"]),
            "evaluated_candidates": int(len(candidate_records)),
            "possible_candidates_per_level": int(total_possible_per_level),
            "refine_levels": int(refine_levels),
        },
        "search_levels": search_levels,
        "mismatch_map": final_level["mismatch_map"],
        "candidates": candidate_records,
        "best_candidate": best,
        "best_mismatch": float(best["mismatch"]),
        "nearest_true_candidate": nearest,
        "score": score,
        "notes": [
            "Grid search assumes a circular anomaly.",
            "Differential mode compares hidden-minus-background residual traces and is usually less fooled by direct arrivals.",
            "Radius and anomaly velocity are fixed from metadata unless CLI overrides are supplied.",
            "The forward model uses a simple Devito acoustic solver with crude zero-style boundaries, not a tuned PML.",
        ],
    }
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction
