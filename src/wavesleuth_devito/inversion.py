"""Simple search-based inversion routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import candidate_axes, grid_extent
from .io import array_string, load_run_npz, save_json, world_from_run
from .metadata import base_metadata
from .scoring import center_error, probability_map_from_mismatch_map, score_circle_reconstruction, trace_mismatch
from .simulation import ForwardTraceEngine
from .uncertainty import candidate_probabilities
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


def _as_positive_values(values: Sequence[float] | None, *, label: str) -> list[float] | None:
    if values is None:
        return None
    cleaned = sorted({round(float(v), 10) for v in values})
    if not cleaned:
        raise ValidationError(f"{label} cannot be empty.")
    if any(v <= 0.0 for v in cleaned):
        raise ValidationError(f"{label} values must be positive.")
    return [float(v) for v in cleaned]


def default_radius_values(base_radius: float) -> list[float]:
    """Return a tiny radius search axis around a reference radius."""
    base = float(base_radius)
    return [0.75 * base, base, 1.25 * base]


def default_velocity_values(background_velocity: float, base_velocity: float) -> list[float]:
    """Return a tiny anomaly-velocity search axis around a reference contrast."""
    bg = float(background_velocity)
    base = float(base_velocity)
    contrast = base - bg
    if abs(contrast) < 1.0e-8:
        return [max(0.1, base * 0.9), base, base * 1.1]
    values = [bg + 0.70 * contrast, base, bg + 1.30 * contrast]
    return [max(0.05, float(v)) for v in values]


def _nearest_true_summary(candidates: list[dict[str, Any]], true_params: dict[str, float], true_velocity: float) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = sorted(candidates, key=lambda c: float(c["mismatch"]))

    def distance(c: dict[str, Any]) -> float:
        cdist = center_error(
            (true_params["center_x"], true_params["center_z"]),
            (float(c["center_x"]), float(c["center_z"])),
        )
        rdist = abs(float(c.get("radius", true_params["radius"])) - float(true_params["radius"]))
        vdist = 0.05 * abs(float(c.get("anomaly_velocity", true_velocity)) - float(true_velocity))
        return cdist + rdist + vdist

    nearest = min(candidates, key=distance)
    rank = 1 + next(i for i, c in enumerate(ranked) if c is nearest)
    return {
        "candidate": nearest,
        "rank_by_mismatch": int(rank),
        "distance_to_true_candidate": float(distance(nearest)),
        "center_distance_to_true": center_error(
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
    radius_values: Sequence[float] | None = None,
    anomaly_velocity_values: Sequence[float] | None = None,
    search_radius: bool = False,
    search_velocity: bool = False,
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
    """Run a grid-search inversion for a circular anomaly.

    v0.3 can search radius and anomaly velocity as small parameter axes in
    addition to center location. The mismatch map shown in visualizations is the
    best mismatch at each center after minimizing over radius/velocity values.
    """
    run = load_run_npz(run_path)
    world = world_from_run(run)
    if anomaly_kind(world) != "circle":
        raise UnsupportedWorldError("The grid-search inversion currently supports circle worlds only.")

    observed = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
    true_params = circle_parameters(world)
    if true_params is None:
        raise UnsupportedWorldError("The observed run does not contain circle metadata.")

    reference_radius = float(radius if radius is not None else true_params["radius"])
    reference_velocity = float(
        anomaly_velocity
        if anomaly_velocity is not None
        else world["medium"].get("anomaly_velocity", world["medium"]["background_velocity"])
    )
    background_velocity = float(world["medium"]["background_velocity"])

    radii = _as_positive_values(radius_values, label="radius_values")
    velocities = _as_positive_values(anomaly_velocity_values, label="anomaly_velocity_values")
    if radii is None:
        radii = default_radius_values(reference_radius) if search_radius else [reference_radius]
    if velocities is None:
        velocities = default_velocity_values(background_velocity, reference_velocity) if search_velocity else [reference_velocity]
    radii = _as_positive_values(radii, label="radius_values") or []
    velocities = _as_positive_values(velocities, label="anomaly_velocity_values") or []

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

    margin = max(max(radii) * 1.10, 0.10 * min(float(world["grid"]["extent_x"]), float(world["grid"]["extent_z"])))
    background_traces: np.ndarray | None = None
    background_forward_runs = 0
    if mismatch_mode == "differential":
        background_model = background_velocity_model_from_world(world)
        background_traces = engine.run(background_model).receiver_traces
        background_forward_runs = 1
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
    total_possible_per_level = candidate_grid_size * candidate_grid_size * len(radii) * len(velocities)
    evaluated_total = 0
    previous_best: dict[str, Any] | None = None
    previous_dx: float | None = None
    previous_dz: float | None = None
    cache: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    stop = False

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
        best_parameter_map: list[list[dict[str, Any] | None]] = [[None for _ in range(len(xs))] for _ in range(len(zs))]
        level_records: list[dict[str, Any]] = []

        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                center_best: dict[str, Any] | None = None
                for cand_radius in radii:
                    for cand_velocity in velocities:
                        if max_candidates is not None and evaluated_total >= int(max_candidates):
                            stop = True
                            break
                        key = (
                            round(float(x), 8),
                            round(float(z), 8),
                            round(float(cand_radius), 8),
                            round(float(cand_velocity), 8),
                        )
                        cached = cache.get(key)
                        if cached is not None:
                            record = cached
                        else:
                            candidate_world = world_with_circle_candidate(
                                world,
                                center_x=float(x),
                                center_z=float(z),
                                radius=float(cand_radius),
                                anomaly_velocity=float(cand_velocity),
                            )
                            velocity_model = velocity_model_from_world(candidate_world)
                            simulated = engine.run(velocity_model).receiver_traces
                            if simulated.shape != observed.shape:
                                raise ValidationError(
                                    f"Candidate traces shape {simulated.shape} does not match observed shape {observed.shape}."
                                )
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
                            record = {
                                "level": int(level),
                                "index_x": int(ix),
                                "index_z": int(iz),
                                "center_x": float(x),
                                "center_z": float(z),
                                "radius": float(cand_radius),
                                "anomaly_velocity": float(cand_velocity),
                                "mismatch": float(mismatch),
                            }
                            candidate_records.append(record)
                            cache[key] = record
                            if not quiet:
                                limit_text = "?" if max_candidates is None else str(max_candidates)
                                print(
                                    f"level {level} candidate {evaluated_total}/{limit_text}: "
                                    f"center=({float(x):.3f}, {float(z):.3f}) "
                                    f"r={float(cand_radius):.3f} v={float(cand_velocity):.3f} mismatch={mismatch:.6g}"
                                )
                        level_records.append(record)
                        if center_best is None or float(record["mismatch"]) < float(center_best["mismatch"]):
                            center_best = record
                    if stop:
                        break
                if center_best is not None:
                    mismatch_map[iz][ix] = float(center_best["mismatch"])
                    best_parameter_map[iz][ix] = center_best
                if stop:
                    break
            if stop:
                break

        if level_records:
            previous_best = select_best_candidate(level_records)
            previous_dx = abs(dx) if dx else previous_dx
            previous_dz = abs(dz) if dz else previous_dz
        uncertainty_summary: dict[str, Any]
        try:
            _prob, uncertainty_summary = probability_map_from_mismatch_map(mismatch_map)
        except ValidationError:
            uncertainty_summary = {}
        search_levels.append(
            {
                "level": int(level),
                "grid_size": int(candidate_grid_size),
                "xs": [float(v) for v in xs],
                "zs": [float(v) for v in zs],
                "radii": [float(v) for v in radii],
                "anomaly_velocities": [float(v) for v in velocities],
                "margin": float(used_margin),
                "mismatch_map": mismatch_map,
                "best_parameter_map": best_parameter_map,
                "best_candidate": previous_best,
                "uncertainty_summary": uncertainty_summary,
                "evaluated_candidates_this_level": int(len({id(r) for r in level_records})),
            }
        )
        if stop:
            break

    best = select_best_candidate(candidate_records)
    score = score_circle_reconstruction(
        world,
        predicted_center_x=float(best["center_x"]),
        predicted_center_z=float(best["center_z"]),
        predicted_radius=float(best["radius"]),
        best_mismatch=float(best["mismatch"]),
    )
    true_velocity = float(world["medium"].get("anomaly_velocity", background_velocity))
    nearest = _nearest_true_summary(candidate_records, true_params, true_velocity)
    final_level = search_levels[-1]
    uncertainty = candidate_probabilities({"candidates": candidate_records})

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
        "search": {
            "search_radius": bool(search_radius or (radius_values is not None)),
            "search_velocity": bool(search_velocity or (anomaly_velocity_values is not None)),
            "radii": [float(v) for v in radii],
            "anomaly_velocities": [float(v) for v in velocities],
        },
        "true_center": {
            "center_x": float(true_params["center_x"]),
            "center_z": float(true_params["center_z"]),
            "radius": float(true_params["radius"]),
            "anomaly_velocity": true_velocity,
        },
        "candidate_grid": {
            "grid_size": int(candidate_grid_size),
            "xs": final_level["xs"],
            "zs": final_level["zs"],
            "radii": [float(v) for v in radii],
            "anomaly_velocities": [float(v) for v in velocities],
            "margin": float(final_level["margin"]),
            "evaluated_candidates": int(len(candidate_records)),
            "forward_runs": int(evaluated_total + background_forward_runs),
            "background_forward_runs": int(background_forward_runs),
            "possible_candidates_per_level": int(total_possible_per_level),
            "refine_levels": int(refine_levels),
        },
        "search_levels": search_levels,
        "mismatch_map": final_level["mismatch_map"],
        "candidates": candidate_records,
        "best_candidate": best,
        "best_mismatch": float(best["mismatch"]),
        "nearest_true_candidate": nearest,
        "uncertainty": uncertainty,
        "score": score,
        "notes": [
            "Grid search assumes a circular anomaly.",
            "Differential mode compares hidden-minus-background residual traces and is usually less fooled by direct arrivals.",
            "v0.3 can scan radius and anomaly velocity, but local refinements currently refine center position only.",
            "The mismatch map minimizes over searched radius/velocity at each candidate center.",
            "The optional sponge boundary is a simple damping layer, not a production PML.",
        ],
    }
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction
