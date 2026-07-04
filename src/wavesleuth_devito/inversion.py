"""Simple search-based inversion routines.

v0.4 keeps the original joint grid search but adds a staged search strategy for
circle inversions where radius and anomaly velocity are unknown. The staged path
first searches for plausible centers with nominal physics, keeps a small top-K
set, then searches radius/velocity locally. This avoids the v0.3 failure mode
where weak, small impostor anomalies could win globally before the center was
well localized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .blind import is_blind_public_world
from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import candidate_axes, grid_extent
from .io import array_string, load_run_npz, save_json, world_from_run
from .metadata import base_metadata
from .scoring import center_error, probability_map_from_mismatch_map, score_circle_reconstruction, score_ellipse_reconstruction, trace_mismatch
from .simulation import ForwardTraceEngine
from .uncertainty import candidate_probabilities
from .world import (
    anomaly_kind,
    background_velocity_model_from_world,
    circle_parameters,
    ellipse_parameters,
    velocity_model_from_world,
    world_with_circle_candidate,
    world_with_ellipse_candidate,
)

SUPPORTED_SEARCH_STRATEGIES = ("auto", "joint", "staged")
SUPPORTED_PARAMETER_PRIORS = ("none", "reference")


def select_best_candidate(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the candidate with the smallest objective mismatch."""
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


def _candidate_key(center_x: float, center_z: float, radius: float, velocity: float) -> tuple[float, float, float, float]:
    return (round(float(center_x), 8), round(float(center_z), 8), round(float(radius), 8), round(float(velocity), 8))


def _parameter_penalty(
    *,
    radius: float,
    velocity: float,
    reference_radius: float,
    reference_velocity: float,
    parameter_prior: str,
    radius_prior_weight: float,
    velocity_prior_weight: float,
) -> float:
    if parameter_prior == "none":
        return 0.0
    if parameter_prior != "reference":
        raise ValidationError("parameter_prior must be 'none' or 'reference'.")
    penalty = 0.0
    if radius_prior_weight:
        scale = max(abs(float(reference_radius)), 1.0e-12)
        penalty += float(radius_prior_weight) * ((float(radius) - float(reference_radius)) / scale) ** 2
    if velocity_prior_weight:
        scale = max(abs(float(reference_velocity)), 1.0e-12)
        penalty += float(velocity_prior_weight) * ((float(velocity) - float(reference_velocity)) / scale) ** 2
    return float(penalty)


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


def _unique_top_centers(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Return the best unique center records by mismatch."""
    seen: set[tuple[float, float]] = set()
    selected: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda c: float(c["mismatch"])):
        key = (round(float(record["center_x"]), 8), round(float(record["center_z"]), 8))
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) >= int(top_k):
            break
    return selected


def _level_summary(
    *,
    level: int,
    stage: str,
    xs: np.ndarray,
    zs: np.ndarray,
    radii: Sequence[float],
    velocities: Sequence[float],
    margin: float,
    records_by_cell: list[list[dict[str, Any] | None]],
    level_records: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    mismatch_map: list[list[float | None]] = []
    best_parameter_map: list[list[dict[str, Any] | None]] = []
    for row in records_by_cell:
        mismatch_row: list[float | None] = []
        param_row: list[dict[str, Any] | None] = []
        for record in row:
            if record is None:
                mismatch_row.append(None)
                param_row.append(None)
            else:
                mismatch_row.append(float(record["mismatch"]))
                param_row.append(record)
        mismatch_map.append(mismatch_row)
        best_parameter_map.append(param_row)
    try:
        _prob, uncertainty_summary = probability_map_from_mismatch_map(mismatch_map)
    except ValidationError:
        uncertainty_summary = {}
    return {
        "level": int(level),
        "stage": stage,
        "grid_size_x": int(len(xs)),
        "grid_size_z": int(len(zs)),
        "xs": [float(v) for v in xs],
        "zs": [float(v) for v in zs],
        "radii": [float(v) for v in radii],
        "anomaly_velocities": [float(v) for v in velocities],
        "margin": float(margin),
        "mismatch_map": mismatch_map,
        "best_parameter_map": best_parameter_map,
        "best_candidate": select_best_candidate(level_records) if level_records else None,
        "uncertainty_summary": uncertainty_summary,
        "evaluated_records_this_level": int(len(level_records)),
        "notes": notes or [],
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
    search_strategy: str = "auto",
    top_k_refine: int = 5,
    final_refine_top_k: int = 1,
    center_metric: str | None = None,
    final_metric: str | None = None,
    parameter_prior: str = "none",
    radius_prior_weight: float = 0.0,
    velocity_prior_weight: float = 0.0,
) -> dict[str, Any]:
    """Run a grid-search inversion for a circular anomaly.

    ``search_strategy='joint'`` reproduces the v0.3 style: every center is
    evaluated with every radius/velocity value. ``search_strategy='staged'`` is
    designed for unknown radius/velocity cases: center first, top-K parameter
    search second, optional local center refinement third.
    """
    run = load_run_npz(run_path)
    world = world_from_run(run)
    answer_hidden = is_blind_public_world(world)
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
    if top_k_refine < 1:
        raise ValidationError("top_k_refine must be at least 1.")
    if final_refine_top_k < 0:
        raise ValidationError("final_refine_top_k must be non-negative.")
    mismatch_mode = mismatch_mode.lower()
    if mismatch_mode not in {"raw", "differential"}:
        raise ValidationError("mismatch_mode must be 'raw' or 'differential'.")
    metric = metric.lower()
    center_metric = (center_metric or metric).lower()
    final_metric = (final_metric or metric).lower()
    search_strategy = search_strategy.lower()
    if search_strategy not in SUPPORTED_SEARCH_STRATEGIES:
        raise ValidationError(f"search_strategy must be one of {SUPPORTED_SEARCH_STRATEGIES}.")
    parameter_prior = parameter_prior.lower()
    if parameter_prior not in SUPPORTED_PARAMETER_PRIORS:
        raise ValidationError("parameter_prior must be 'none' or 'reference'.")
    if radius_prior_weight < 0.0 or velocity_prior_weight < 0.0:
        raise ValidationError("parameter prior weights must be non-negative.")

    axis_searching = len(radii) > 1 or len(velocities) > 1
    if search_strategy == "auto":
        used_strategy = "staged" if axis_searching else "joint"
    else:
        used_strategy = search_strategy

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
    trace_cache: dict[tuple[float, float, float, float], np.ndarray] = {}
    forward_candidate_runs = 0
    scored_candidates = 0
    stop = False

    def evaluate_candidate(
        *,
        center_x: float,
        center_z: float,
        cand_radius: float,
        cand_velocity: float,
        level: int,
        stage: str,
        ix: int,
        iz: int,
        used_metric: str,
        center_rank: int | None = None,
    ) -> dict[str, Any] | None:
        nonlocal forward_candidate_runs, scored_candidates, stop
        if max_candidates is not None and scored_candidates >= int(max_candidates):
            stop = True
            return None
        key = _candidate_key(center_x, center_z, cand_radius, cand_velocity)
        simulated_target = trace_cache.get(key)
        if simulated_target is None:
            candidate_world = world_with_circle_candidate(
                world,
                center_x=float(center_x),
                center_z=float(center_z),
                radius=float(cand_radius),
                anomaly_velocity=float(cand_velocity),
            )
            velocity_model = velocity_model_from_world(candidate_world)
            simulated = engine.run(velocity_model).receiver_traces
            if simulated.shape != observed.shape:
                raise ValidationError(f"Candidate traces shape {simulated.shape} does not match observed shape {observed.shape}.")
            simulated_target = simulated - background_traces if background_traces is not None else simulated
            trace_cache[key] = simulated_target
            forward_candidate_runs += 1
        data_mismatch = trace_mismatch(
            observed_target,
            simulated_target,
            metric=used_metric,
            time=time,
            time_min=time_min,
            time_max=time_max,
            normalize_traces=normalize_traces,
        )
        prior_penalty = _parameter_penalty(
            radius=float(cand_radius),
            velocity=float(cand_velocity),
            reference_radius=reference_radius,
            reference_velocity=reference_velocity,
            parameter_prior=parameter_prior,
            radius_prior_weight=radius_prior_weight,
            velocity_prior_weight=velocity_prior_weight,
        )
        mismatch = float(data_mismatch) + float(prior_penalty)
        scored_candidates += 1
        record = {
            "level": int(level),
            "stage": stage,
            "index_x": int(ix),
            "index_z": int(iz),
            "center_x": float(center_x),
            "center_z": float(center_z),
            "radius": float(cand_radius),
            "anomaly_velocity": float(cand_velocity),
            "metric": used_metric,
            "data_mismatch": float(data_mismatch),
            "prior_penalty": float(prior_penalty),
            "mismatch": mismatch,
        }
        if center_rank is not None:
            record["center_rank"] = int(center_rank)
        candidate_records.append(record)
        if not quiet:
            limit_text = "?" if max_candidates is None else str(max_candidates)
            print(
                f"{stage} candidate {scored_candidates}/{limit_text}: "
                f"center=({float(center_x):.3f}, {float(center_z):.3f}) "
                f"r={float(cand_radius):.3f} v={float(cand_velocity):.3f} "
                f"metric={used_metric} mismatch={mismatch:.6g}"
            )
        return record

    def evaluate_grid_level(
        *,
        level: int,
        stage: str,
        xs: np.ndarray,
        zs: np.ndarray,
        level_radii: Sequence[float],
        level_velocities: Sequence[float],
        level_metric: str,
        used_margin: float,
        center_rank: int | None = None,
        notes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records_by_cell: list[list[dict[str, Any] | None]] = [[None for _ in range(len(xs))] for _ in range(len(zs))]
        level_records: list[dict[str, Any]] = []
        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                center_best: dict[str, Any] | None = None
                for cand_radius in level_radii:
                    for cand_velocity in level_velocities:
                        record = evaluate_candidate(
                            center_x=float(x),
                            center_z=float(z),
                            cand_radius=float(cand_radius),
                            cand_velocity=float(cand_velocity),
                            level=level,
                            stage=stage,
                            ix=ix,
                            iz=iz,
                            used_metric=level_metric,
                            center_rank=center_rank,
                        )
                        if record is None:
                            break
                        level_records.append(record)
                        if center_best is None or float(record["mismatch"]) < float(center_best["mismatch"]):
                            center_best = record
                    if stop:
                        break
                records_by_cell[iz][ix] = center_best
                if stop:
                    break
            if stop:
                break
        search_levels.append(
            _level_summary(
                level=level,
                stage=stage,
                xs=xs,
                zs=zs,
                radii=level_radii,
                velocities=level_velocities,
                margin=used_margin,
                records_by_cell=records_by_cell,
                level_records=level_records,
                notes=notes,
            )
        )
        return level_records

    if used_strategy == "joint":
        previous_best: dict[str, Any] | None = None
        previous_dx: float | None = None
        previous_dz: float | None = None
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
            dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else previous_dx or 0.0
            dz = abs(float(zs[1] - zs[0])) if len(zs) > 1 else previous_dz or 0.0
            level_records = evaluate_grid_level(
                level=level,
                stage="joint",
                xs=xs,
                zs=zs,
                level_radii=radii,
                level_velocities=velocities,
                level_metric=metric,
                used_margin=used_margin,
                notes=["v0.3-compatible joint center/radius/velocity search."],
            )
            if level_records:
                previous_best = select_best_candidate(level_records)
                previous_dx = dx
                previous_dz = dz
            if stop:
                break
    else:
        xs, zs, used_margin = _candidate_grid_around(world, grid_size=candidate_grid_size, margin=margin)
        dx0 = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 0.0
        dz0 = abs(float(zs[1] - zs[0])) if len(zs) > 1 else 0.0
        center_records = evaluate_grid_level(
            level=0,
            stage="coarse-center",
            xs=xs,
            zs=zs,
            level_radii=[reference_radius],
            level_velocities=[reference_velocity],
            level_metric=center_metric,
            used_margin=used_margin,
            notes=["Center-only pass with nominal radius and velocity."],
        )
        center_pool = list(center_records)
        current_dx, current_dz = dx0, dz0

        for refine in range(int(refine_levels)):
            if stop:
                break
            parents = _unique_top_centers(center_pool, top_k_refine)
            refined_records: list[dict[str, Any]] = []
            for rank, parent in enumerate(parents, start=1):
                local_xs, local_zs, local_margin = _candidate_grid_around(
                    world,
                    grid_size=candidate_grid_size,
                    margin=margin,
                    center_x=float(parent["center_x"]),
                    center_z=float(parent["center_z"]),
                    span_x=current_dx,
                    span_z=current_dz,
                )
                local_records = evaluate_grid_level(
                    level=1 + refine,
                    stage="topk-center-refine",
                    xs=local_xs,
                    zs=local_zs,
                    level_radii=[reference_radius],
                    level_velocities=[reference_velocity],
                    level_metric=center_metric,
                    used_margin=local_margin,
                    center_rank=rank,
                    notes=[f"Local center refinement around top-K parent rank {rank} from the previous center pool."],
                )
                refined_records.extend(local_records)
                if stop:
                    break
            if refined_records:
                center_pool.extend(refined_records)
                current_dx = current_dx / max(candidate_grid_size - 1, 1)
                current_dz = current_dz / max(candidate_grid_size - 1, 1)

        if not stop:
            top_centers = _unique_top_centers(center_pool, top_k_refine)
            parameter_records: list[dict[str, Any]] = []
            for rank, center in enumerate(top_centers, start=1):
                records_by_cell = [[None]]
                local_records: list[dict[str, Any]] = []
                for cand_radius in radii:
                    for cand_velocity in velocities:
                        record = evaluate_candidate(
                            center_x=float(center["center_x"]),
                            center_z=float(center["center_z"]),
                            cand_radius=float(cand_radius),
                            cand_velocity=float(cand_velocity),
                            level=100,
                            stage="topk-parameter",
                            ix=0,
                            iz=0,
                            used_metric=metric,
                            center_rank=rank,
                        )
                        if record is None:
                            break
                        local_records.append(record)
                        parameter_records.append(record)
                        if records_by_cell[0][0] is None or float(record["mismatch"]) < float(records_by_cell[0][0]["mismatch"]):
                            records_by_cell[0][0] = record
                    if stop:
                        break
                search_levels.append(
                    _level_summary(
                        level=100,
                        stage="topk-parameter",
                        xs=np.asarray([float(center["center_x"])], dtype=np.float32),
                        zs=np.asarray([float(center["center_z"])], dtype=np.float32),
                        radii=radii,
                        velocities=velocities,
                        margin=margin,
                        records_by_cell=records_by_cell,
                        level_records=local_records,
                        notes=[f"Radius/velocity search at refined center rank {rank}."]
                    )
                )
                if stop:
                    break

            if parameter_records and final_refine_top_k > 0 and refine_levels > 0 and not stop:
                final_parents = sorted(parameter_records, key=lambda c: float(c["mismatch"]))[: int(final_refine_top_k)]
                final_span_x = max(current_dx, dx0 / max(candidate_grid_size - 1, 1))
                final_span_z = max(current_dz, dz0 / max(candidate_grid_size - 1, 1))
                for rank, parent in enumerate(final_parents, start=1):
                    final_xs, final_zs, final_margin = _candidate_grid_around(
                        world,
                        grid_size=candidate_grid_size,
                        margin=margin,
                        center_x=float(parent["center_x"]),
                        center_z=float(parent["center_z"]),
                        span_x=final_span_x,
                        span_z=final_span_z,
                    )
                    evaluate_grid_level(
                        level=200,
                        stage="final-center-refine",
                        xs=final_xs,
                        zs=final_zs,
                        level_radii=[float(parent["radius"])],
                        level_velocities=[float(parent["anomaly_velocity"])],
                        level_metric=final_metric,
                        used_margin=final_margin,
                        center_rank=rank,
                        notes=["Final local center refinement using the best radius/velocity candidate."],
                    )
                    if stop:
                        break

    if not candidate_records:
        raise ValidationError("No candidates were evaluated.")

    if used_strategy == "staged" and axis_searching:
        preferred = [r for r in candidate_records if r.get("stage") in {"topk-parameter", "final-center-refine"}]
        best_pool = preferred or candidate_records
    else:
        best_pool = candidate_records
    best = select_best_candidate(best_pool)

    true_velocity = float(world["medium"].get("anomaly_velocity", background_velocity))
    if answer_hidden:
        score = {"supported": False, "answer_hidden": True, "message": "Blind public metadata hides the answer; score with the secret world or score-challenge."}
        nearest = {}
        true_center_payload = None
    else:
        score = score_circle_reconstruction(
            world,
            predicted_center_x=float(best["center_x"]),
            predicted_center_z=float(best["center_z"]),
            predicted_radius=float(best["radius"]),
            best_mismatch=float(best["mismatch"]),
        )
        nearest = _nearest_true_summary(candidate_records, true_params, true_velocity)
        true_center_payload = {"center_x": float(true_params["center_x"]), "center_z": float(true_params["center_z"]), "radius": float(true_params["radius"]), "anomaly_velocity": true_velocity}
    final_level = search_levels[-1]
    uncertainty = candidate_probabilities({"candidates": candidate_records})

    total_possible_per_joint_level = candidate_grid_size * candidate_grid_size * len(radii) * len(velocities)
    reconstruction: dict[str, Any] = {
        **base_metadata(),
        "method": "grid-search",
        "run_path": str(run_path),
        "world_name": world.get("name", "unknown"),
        "world": world,
        "objective": {
            "mismatch_mode": mismatch_mode,
            "metric": metric,
            "center_metric": center_metric,
            "final_metric": final_metric,
            "time_min": None if time_min is None else float(time_min),
            "time_max": None if time_max is None else float(time_max),
            "normalize_traces": bool(normalize_traces),
            "shot_mode": used_shot_mode,
            "background_subtracted": bool(mismatch_mode == "differential"),
            "parameter_prior": parameter_prior,
            "radius_prior_weight": float(radius_prior_weight),
            "velocity_prior_weight": float(velocity_prior_weight),
            "mismatch_includes_prior_penalty": bool(parameter_prior != "none" and (radius_prior_weight > 0.0 or velocity_prior_weight > 0.0)),
        },
        "search": {
            "search_strategy": used_strategy,
            "requested_search_strategy": search_strategy,
            "search_radius": bool(search_radius or (radius_values is not None)),
            "search_velocity": bool(search_velocity or (anomaly_velocity_values is not None)),
            "radii": [float(v) for v in radii],
            "anomaly_velocities": [float(v) for v in velocities],
            "top_k_refine": int(top_k_refine),
            "final_refine_top_k": int(final_refine_top_k),
            "staged_axis_search": bool(axis_searching and used_strategy == "staged"),
        },
        "true_center": true_center_payload,
        "answer_hidden": bool(answer_hidden),
        "candidate_grid": {
            "grid_size": int(candidate_grid_size),
            "xs": final_level["xs"],
            "zs": final_level["zs"],
            "radii": [float(v) for v in radii],
            "anomaly_velocities": [float(v) for v in velocities],
            "margin": float(final_level["margin"]),
            "evaluated_candidates": int(scored_candidates),
            "unique_forward_candidates": int(forward_candidate_runs),
            "forward_runs": int(forward_candidate_runs + background_forward_runs),
            "background_forward_runs": int(background_forward_runs),
            "possible_candidates_per_joint_level": int(total_possible_per_joint_level),
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
            "v0.4 adds staged search: center first, top-K radius/velocity search second, optional final local center refinement.",
            "Joint strategy remains available for v0.3-style behavior and for diagnosing impostor solutions.",
            "The mismatch map is taken from the final displayed search level; inspect search_levels for the full multi-stage trajectory.",
            "The optional sponge boundary is a simple damping layer, not a production PML.",
        ],
    }
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction


def default_axis_values(base_axis: float) -> list[float]:
    """Return a tiny ellipse-axis search axis around a reference semi-axis."""
    base = float(base_axis)
    return [0.85 * base, base, 1.15 * base]


def default_angle_values(base_angle_degrees: float) -> list[float]:
    """Return a tiny angle search axis around a reference ellipse orientation."""
    base = float(base_angle_degrees)
    return [base - 20.0, base, base + 20.0]


def _ellipse_candidate_key(
    center_x: float,
    center_z: float,
    radius_x: float,
    radius_z: float,
    angle_degrees: float,
    velocity: float,
) -> tuple[float, float, float, float, float, float]:
    return (
        round(float(center_x), 8),
        round(float(center_z), 8),
        round(float(radius_x), 8),
        round(float(radius_z), 8),
        round(float(angle_degrees), 8),
        round(float(velocity), 8),
    )


def _ellipse_level_summary(
    *,
    level: int,
    stage: str,
    xs: np.ndarray,
    zs: np.ndarray,
    radius_x_values: Sequence[float],
    radius_z_values: Sequence[float],
    angle_values: Sequence[float],
    velocities: Sequence[float],
    margin: float,
    records_by_cell: list[list[dict[str, Any] | None]],
    level_records: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    mismatch_map: list[list[float | None]] = []
    best_parameter_map: list[list[dict[str, Any] | None]] = []
    for row in records_by_cell:
        mismatch_row: list[float | None] = []
        param_row: list[dict[str, Any] | None] = []
        for record in row:
            if record is None:
                mismatch_row.append(None)
                param_row.append(None)
            else:
                mismatch_row.append(float(record["mismatch"]))
                param_row.append(record)
        mismatch_map.append(mismatch_row)
        best_parameter_map.append(param_row)
    try:
        _prob, uncertainty_summary = probability_map_from_mismatch_map(mismatch_map)
    except ValidationError:
        uncertainty_summary = {}
    return {
        "level": int(level),
        "stage": stage,
        "grid_size_x": int(len(xs)),
        "grid_size_z": int(len(zs)),
        "xs": [float(v) for v in xs],
        "zs": [float(v) for v in zs],
        "radius_x_values": [float(v) for v in radius_x_values],
        "radius_z_values": [float(v) for v in radius_z_values],
        "angle_values": [float(v) for v in angle_values],
        "anomaly_velocities": [float(v) for v in velocities],
        "margin": float(margin),
        "mismatch_map": mismatch_map,
        "best_parameter_map": best_parameter_map,
        "best_candidate": select_best_candidate(level_records) if level_records else None,
        "uncertainty_summary": uncertainty_summary,
        "evaluated_records_this_level": int(len(level_records)),
        "notes": notes or [],
    }


def _nearest_true_ellipse_summary(candidates: list[dict[str, Any]], true_params: dict[str, float], true_velocity: float) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = sorted(candidates, key=lambda c: float(c["mismatch"]))

    def distance(c: dict[str, Any]) -> float:
        cdist = center_error(
            (true_params["center_x"], true_params["center_z"]),
            (float(c["center_x"]), float(c["center_z"])),
        )
        rx = abs(float(c.get("radius_x", true_params["radius_x"])) - float(true_params["radius_x"]))
        rz = abs(float(c.get("radius_z", true_params["radius_z"])) - float(true_params["radius_z"]))
        angle = abs(((float(c.get("angle_degrees", true_params["angle_degrees"])) - float(true_params["angle_degrees"]) + 90.0) % 180.0) - 90.0) / 180.0
        vdist = 0.05 * abs(float(c.get("anomaly_velocity", true_velocity)) - float(true_velocity))
        return cdist + rx + rz + angle + vdist

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


def grid_search_ellipse(
    run_path: str | Path,
    *,
    out_path: str | Path | None = None,
    candidate_grid_size: int = 5,
    radius_x: float | None = None,
    radius_z: float | None = None,
    angle_degrees: float | None = None,
    anomaly_velocity: float | None = None,
    radius_x_values: Sequence[float] | None = None,
    radius_z_values: Sequence[float] | None = None,
    angle_values: Sequence[float] | None = None,
    anomaly_velocity_values: Sequence[float] | None = None,
    search_axes: bool = False,
    search_angle: bool = False,
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
    """Grid-search inversion for an elliptical anomaly.

    v0.5 intentionally keeps ellipse inversion conservative: by default it
    searches the ellipse center while holding semi-axes, angle, and velocity from
    metadata. Optional small axes/angle/velocity searches are provided for
    experiments, but the first non-circle baseline is center recovery.
    """
    run = load_run_npz(run_path)
    world = world_from_run(run)
    answer_hidden = is_blind_public_world(world)
    if anomaly_kind(world) != "ellipse":
        raise UnsupportedWorldError("ellipse-grid-search currently supports ellipse worlds only.")

    observed = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
    true_params = ellipse_parameters(world)
    if true_params is None:
        raise UnsupportedWorldError("The observed run does not contain ellipse metadata.")

    reference_radius_x = float(radius_x if radius_x is not None else true_params["radius_x"])
    reference_radius_z = float(radius_z if radius_z is not None else true_params["radius_z"])
    reference_angle = float(angle_degrees if angle_degrees is not None else true_params["angle_degrees"])
    reference_velocity = float(
        anomaly_velocity
        if anomaly_velocity is not None
        else world["medium"].get("anomaly_velocity", world["medium"]["background_velocity"])
    )
    background_velocity = float(world["medium"]["background_velocity"])

    rx_values = _as_positive_values(radius_x_values, label="radius_x_values")
    rz_values = _as_positive_values(radius_z_values, label="radius_z_values")
    velocities = _as_positive_values(anomaly_velocity_values, label="anomaly_velocity_values")
    if rx_values is None:
        rx_values = default_axis_values(reference_radius_x) if search_axes else [reference_radius_x]
    if rz_values is None:
        rz_values = default_axis_values(reference_radius_z) if search_axes else [reference_radius_z]
    if angle_values is None:
        angles = default_angle_values(reference_angle) if search_angle else [reference_angle]
    else:
        angles = sorted({round(float(v), 10) for v in angle_values})
        if not angles:
            raise ValidationError("angle_values cannot be empty.")
    if velocities is None:
        velocities = default_velocity_values(background_velocity, reference_velocity) if search_velocity else [reference_velocity]
    rx_values = _as_positive_values(rx_values, label="radius_x_values") or []
    rz_values = _as_positive_values(rz_values, label="radius_z_values") or []
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
    metric = metric.lower()

    used_shot_mode = shot_mode or _shot_mode_from_run(run, observed)
    if used_shot_mode not in {"simultaneous", "sequential"}:
        raise ValidationError("shot_mode must be 'simultaneous' or 'sequential'.")

    engine = ForwardTraceEngine(world, shot_mode=used_shot_mode, save_wavefield=False, quiet=quiet)
    margin = max(max(rx_values + rz_values) * 1.15, 0.10 * min(float(world["grid"]["extent_x"]), float(world["grid"]["extent_z"])))

    background_traces: np.ndarray | None = None
    background_forward_runs = 0
    if mismatch_mode == "differential":
        background_model = background_velocity_model_from_world(world)
        background_traces = engine.run(background_model).receiver_traces
        background_forward_runs = 1
        if background_traces.shape != observed.shape:
            raise ValidationError(f"Background traces shape {background_traces.shape} does not match observed shape {observed.shape}.")
        observed_target = observed - background_traces
    else:
        observed_target = observed

    candidate_records: list[dict[str, Any]] = []
    search_levels: list[dict[str, Any]] = []
    trace_cache: dict[tuple[float, float, float, float, float, float], np.ndarray] = {}
    forward_candidate_runs = 0
    scored_candidates = 0
    stop = False

    def evaluate_candidate(
        *,
        center_x: float,
        center_z: float,
        cand_radius_x: float,
        cand_radius_z: float,
        cand_angle: float,
        cand_velocity: float,
        level: int,
        stage: str,
        ix: int,
        iz: int,
    ) -> dict[str, Any] | None:
        nonlocal forward_candidate_runs, scored_candidates, stop
        if max_candidates is not None and scored_candidates >= int(max_candidates):
            stop = True
            return None
        key = _ellipse_candidate_key(center_x, center_z, cand_radius_x, cand_radius_z, cand_angle, cand_velocity)
        simulated_target = trace_cache.get(key)
        if simulated_target is None:
            candidate_world = world_with_ellipse_candidate(
                world,
                center_x=float(center_x),
                center_z=float(center_z),
                radius_x=float(cand_radius_x),
                radius_z=float(cand_radius_z),
                angle_degrees=float(cand_angle),
                anomaly_velocity=float(cand_velocity),
            )
            velocity_model = velocity_model_from_world(candidate_world)
            simulated = engine.run(velocity_model).receiver_traces
            if simulated.shape != observed.shape:
                raise ValidationError(f"Candidate traces shape {simulated.shape} does not match observed shape {observed.shape}.")
            simulated_target = simulated - background_traces if background_traces is not None else simulated
            trace_cache[key] = simulated_target
            forward_candidate_runs += 1
        mismatch = trace_mismatch(
            observed_target,
            simulated_target,
            metric=metric,
            time=time,
            time_min=time_min,
            time_max=time_max,
            normalize_traces=normalize_traces,
        )
        scored_candidates += 1
        record = {
            "kind": "ellipse",
            "level": int(level),
            "stage": stage,
            "index_x": int(ix),
            "index_z": int(iz),
            "center_x": float(center_x),
            "center_z": float(center_z),
            "radius_x": float(cand_radius_x),
            "radius_z": float(cand_radius_z),
            "angle_degrees": float(cand_angle),
            "anomaly_velocity": float(cand_velocity),
            "metric": metric,
            "data_mismatch": float(mismatch),
            "mismatch": float(mismatch),
        }
        candidate_records.append(record)
        if not quiet:
            limit_text = "?" if max_candidates is None else str(max_candidates)
            print(
                f"ellipse candidate {scored_candidates}/{limit_text}: "
                f"center=({float(center_x):.3f}, {float(center_z):.3f}) "
                f"rx={float(cand_radius_x):.3f} rz={float(cand_radius_z):.3f} "
                f"angle={float(cand_angle):.1f} v={float(cand_velocity):.3f} mismatch={mismatch:.6g}"
            )
        return record

    previous_best: dict[str, Any] | None = None
    previous_dx: float | None = None
    previous_dz: float | None = None

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
        dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else previous_dx or 0.0
        dz = abs(float(zs[1] - zs[0])) if len(zs) > 1 else previous_dz or 0.0
        records_by_cell: list[list[dict[str, Any] | None]] = [[None for _ in range(len(xs))] for _ in range(len(zs))]
        level_records: list[dict[str, Any]] = []
        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                center_best: dict[str, Any] | None = None
                for cand_rx in rx_values:
                    for cand_rz in rz_values:
                        for cand_angle in angles:
                            for cand_velocity in velocities:
                                record = evaluate_candidate(
                                    center_x=float(x),
                                    center_z=float(z),
                                    cand_radius_x=float(cand_rx),
                                    cand_radius_z=float(cand_rz),
                                    cand_angle=float(cand_angle),
                                    cand_velocity=float(cand_velocity),
                                    level=level,
                                    stage="ellipse" if level == 0 else "ellipse-refine",
                                    ix=ix,
                                    iz=iz,
                                )
                                if record is None:
                                    break
                                level_records.append(record)
                                if center_best is None or float(record["mismatch"]) < float(center_best["mismatch"]):
                                    center_best = record
                            if stop:
                                break
                        if stop:
                            break
                    if stop:
                        break
                records_by_cell[iz][ix] = center_best
                if stop:
                    break
            if stop:
                break
        search_levels.append(
            _ellipse_level_summary(
                level=level,
                stage="ellipse" if level == 0 else "ellipse-refine",
                xs=xs,
                zs=zs,
                radius_x_values=rx_values,
                radius_z_values=rz_values,
                angle_values=angles,
                velocities=velocities,
                margin=used_margin,
                records_by_cell=records_by_cell,
                level_records=level_records,
                notes=["v0.5 ellipse grid search; by default only center is unknown."],
            )
        )
        if level_records:
            previous_best = select_best_candidate(level_records)
            previous_dx = dx
            previous_dz = dz
        if stop:
            break

    if not candidate_records:
        raise ValidationError("No ellipse candidates were evaluated.")

    best = select_best_candidate(candidate_records)
    true_velocity = float(world["medium"].get("anomaly_velocity", background_velocity))
    if answer_hidden:
        score = {"supported": False, "answer_hidden": True, "message": "Blind public metadata hides the answer; score with the secret world or score-challenge."}
        nearest = {}
        true_center_payload = None
    else:
        score = score_ellipse_reconstruction(
            world,
            predicted_center_x=float(best["center_x"]),
            predicted_center_z=float(best["center_z"]),
            predicted_radius_x=float(best["radius_x"]),
            predicted_radius_z=float(best["radius_z"]),
            predicted_angle_degrees=float(best["angle_degrees"]),
            predicted_anomaly_velocity=float(best["anomaly_velocity"]),
            best_mismatch=float(best["mismatch"]),
        )
        nearest = _nearest_true_ellipse_summary(candidate_records, true_params, true_velocity)
        true_center_payload = {"center_x": float(true_params["center_x"]), "center_z": float(true_params["center_z"]), "radius_x": float(true_params["radius_x"]), "radius_z": float(true_params["radius_z"]), "angle_degrees": float(true_params["angle_degrees"]), "anomaly_velocity": true_velocity}
    final_level = search_levels[-1]
    uncertainty = candidate_probabilities({"candidates": candidate_records})

    reconstruction: dict[str, Any] = {
        **base_metadata(),
        "method": "ellipse-grid-search",
        "target_kind": "ellipse",
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
            "search_strategy": "ellipse-grid",
            "search_axes": bool(search_axes or radius_x_values is not None or radius_z_values is not None),
            "search_angle": bool(search_angle or angle_values is not None),
            "search_velocity": bool(search_velocity or anomaly_velocity_values is not None),
            "radius_x_values": [float(v) for v in rx_values],
            "radius_z_values": [float(v) for v in rz_values],
            "angle_values": [float(v) for v in angles],
            "anomaly_velocities": [float(v) for v in velocities],
        },
        "true_center": true_center_payload,
        "answer_hidden": bool(answer_hidden),
        "candidate_grid": {
            "grid_size": int(candidate_grid_size),
            "xs": final_level["xs"],
            "zs": final_level["zs"],
            "radius_x_values": [float(v) for v in rx_values],
            "radius_z_values": [float(v) for v in rz_values],
            "angle_values": [float(v) for v in angles],
            "anomaly_velocities": [float(v) for v in velocities],
            "margin": float(final_level["margin"]),
            "evaluated_candidates": int(scored_candidates),
            "unique_forward_candidates": int(forward_candidate_runs),
            "forward_runs": int(forward_candidate_runs + background_forward_runs),
            "background_forward_runs": int(background_forward_runs),
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
            "v0.5 adds the first non-circle inversion path: ellipse-grid-search.",
            "The conservative default searches ellipse center while holding axes, angle, and velocity from metadata.",
            "Optional axes/angle/velocity searches are provided for experiments, but can be ambiguous under sparse data.",
        ],
    }
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction

