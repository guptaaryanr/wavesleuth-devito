"""Coarse block-mask worlds and greedy mask inversion for WaveSleuth-Devito.

v0.8 is the first step from parametric objects toward image-like reconstruction.
The implementation is intentionally small: divide the domain into a coarse grid,
try adding cells, and verify each candidate with the Devito forward model.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .exceptions import ValidationError
from .geometry import coordinate_mesh, grid_extent, grid_shape
from .io import array_string, ensure_parent, load_json, load_run_npz, save_json, world_from_run
from .metadata import base_metadata
from .scoring import iou_score, trace_mismatch
from .simulation import ForwardTraceEngine
from .world import anomaly_kind, anomaly_mask_from_world, background_velocity_model_from_world, validate_world, velocity_model_from_world


def active_cell_tuples(cells: Iterable[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return sorted unique ``(i, j)`` cell coordinates."""
    out = sorted({(int(cell.get("i", cell.get("ix"))), int(cell.get("j", cell.get("iz")))) for cell in cells})
    return out


def json_cells(cells: Iterable[dict[str, Any] | tuple[int, int]]) -> list[dict[str, int]]:
    """Return stable JSON cell dictionaries with keys ``i`` and ``j``."""
    cleaned: set[tuple[int, int]] = set()
    for cell in cells:
        if isinstance(cell, dict):
            cleaned.add((int(cell.get("i", cell.get("ix"))), int(cell.get("j", cell.get("iz")))))
        else:
            i, j = cell
            cleaned.add((int(i), int(j)))
    return [{"i": int(i), "j": int(j)} for i, j in sorted(cleaned)]


def default_mask_block_cells() -> list[dict[str, int]]:
    """Return the deterministic v0.8 hidden block mask."""
    return json_cells([(2, 2), (3, 2), (3, 3), (4, 3), (2, 4)])


def cell_bounds(world: dict[str, Any], i: int, j: int, *, cell_grid_size: int | None = None) -> tuple[float, float, float, float]:
    """Return physical bounds ``(x0, x1, z0, z1)`` for a coarse cell."""
    n = int(cell_grid_size or world.get("medium", {}).get("anomaly", {}).get("cell_grid_size", 6))
    if n < 2:
        raise ValidationError("cell_grid_size must be at least 2.")
    extent_x, extent_z = grid_extent(world)
    ii = int(i)
    jj = int(j)
    if not (0 <= ii < n and 0 <= jj < n):
        raise ValidationError(f"Cell ({ii}, {jj}) outside 0..{n - 1}.")
    return extent_x * ii / n, extent_x * (ii + 1) / n, extent_z * jj / n, extent_z * (jj + 1) / n


def cell_records(world: dict[str, Any], *, cell_grid_size: int = 6) -> list[dict[str, Any]]:
    """Return every coarse candidate cell with physical bounds and center."""
    n = int(cell_grid_size)
    records: list[dict[str, Any]] = []
    for j in range(n):
        for i in range(n):
            x0, x1, z0, z1 = cell_bounds(world, i, j, cell_grid_size=n)
            records.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "x_min": float(x0),
                    "x_max": float(x1),
                    "z_min": float(z0),
                    "z_max": float(z1),
                    "center_x": float(0.5 * (x0 + x1)),
                    "center_z": float(0.5 * (z0 + z1)),
                }
            )
    return records


def mask_blocks_mask_from_cells(world: dict[str, Any], cells: Iterable[dict[str, Any]], *, cell_grid_size: int | None = None) -> np.ndarray:
    """Return a fine-grid boolean mask from active coarse cells."""
    xmesh, zmesh = coordinate_mesh(world)
    mask = np.zeros(grid_shape(world), dtype=bool)
    n = int(cell_grid_size or world.get("medium", {}).get("anomaly", {}).get("cell_grid_size", 6))
    for i, j in active_cell_tuples(cells):
        x0, x1, z0, z1 = cell_bounds(world, i, j, cell_grid_size=n)
        mask |= (xmesh >= x0) & (xmesh <= x1) & (zmesh >= z0) & (zmesh <= z1)
    return mask


def mask_centroid(world: dict[str, Any], mask: np.ndarray) -> dict[str, float | None]:
    """Return physical centroid of a boolean mask."""
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        return {"center_x": None, "center_z": None}
    xmesh, zmesh = coordinate_mesh(world)
    return {"center_x": float(np.mean(xmesh[arr])), "center_z": float(np.mean(zmesh[arr]))}


def world_with_mask_blocks_candidate(
    world: dict[str, Any],
    *,
    active_cells: Iterable[dict[str, Any] | tuple[int, int]],
    anomaly_velocity: float | None = None,
    cell_grid_size: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``world`` with a block-mask candidate anomaly."""
    candidate = copy.deepcopy(world)
    n = int(cell_grid_size or world.get("medium", {}).get("anomaly", {}).get("cell_grid_size", 6))
    velocity = float(anomaly_velocity if anomaly_velocity is not None else candidate.get("medium", {}).get("anomaly_velocity", 2.2))
    candidate["name"] = name or "candidate_mask_blocks"
    candidate.setdefault("medium", {})["anomaly_velocity"] = velocity
    candidate["medium"]["anomaly"] = {
        "kind": "mask-blocks",
        "cell_grid_size": int(n),
        "active_cells": json_cells(active_cells),
        "cell_velocity": velocity,
    }
    validate_world(candidate)
    return candidate


def velocity_model_from_mask_blocks_candidate(
    world: dict[str, Any],
    *,
    active_cells: Iterable[dict[str, Any] | tuple[int, int]],
    anomaly_velocity: float,
    cell_grid_size: int,
) -> np.ndarray:
    """Return velocity model for a block-mask candidate."""
    candidate = world_with_mask_blocks_candidate(
        world,
        active_cells=active_cells,
        anomaly_velocity=anomaly_velocity,
        cell_grid_size=cell_grid_size,
    )
    return velocity_model_from_world(candidate)


def predicted_mask_from_reconstruction(true_world: dict[str, Any], reconstruction: dict[str, Any]) -> np.ndarray:
    """Return the predicted fine-grid mask encoded by a cell-search reconstruction."""
    best = reconstruction.get("best_candidate", {}) if isinstance(reconstruction, dict) else {}
    if not isinstance(best, dict):
        raise ValidationError("Cell reconstruction missing best_candidate.")
    cells = best.get("active_cells", best.get("selected_cells", best.get("cells", [])))
    n = int(best.get("cell_grid_size", reconstruction.get("cell_grid_size", 0)) or 0)
    if n < 2:
        n = int((reconstruction.get("candidate_grid", {}) or {}).get("cell_grid_size", 0) or 0)
    if n < 2:
        n = int(true_world.get("medium", {}).get("anomaly", {}).get("cell_grid_size", 6))
    return mask_blocks_mask_from_cells(true_world, cells, cell_grid_size=n)


def score_mask_blocks_reconstruction(true_world: dict[str, Any], reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Score a block-mask reconstruction using mask IoU and centroid error."""
    true_mask = anomaly_mask_from_world(true_world)
    pred_mask = predicted_mask_from_reconstruction(true_world, reconstruction)
    iou = iou_score(true_mask, pred_mask)
    true_centroid = mask_centroid(true_world, true_mask)
    pred_centroid = mask_centroid(true_world, pred_mask)
    center_error = None
    normalized_center_error = None
    if true_centroid["center_x"] is not None and pred_centroid["center_x"] is not None:
        dx = float(true_centroid["center_x"]) - float(pred_centroid["center_x"])
        dz = float(true_centroid["center_z"]) - float(pred_centroid["center_z"])
        center_error = float((dx * dx + dz * dz) ** 0.5)
        extent_x, extent_z = grid_extent(true_world)
        normalized_center_error = center_error / max(float((extent_x * extent_x + extent_z * extent_z) ** 0.5), 1.0e-12)
    best = reconstruction.get("best_candidate", {}) if isinstance(reconstruction, dict) else {}
    true_cells = true_world.get("medium", {}).get("anomaly", {}).get("active_cells", [])
    pred_cells = best.get("active_cells", best.get("selected_cells", [])) if isinstance(best, dict) else []
    result: dict[str, Any] = {
        "supported": True,
        "target_kind": "mask-blocks",
        "true_world_kind": anomaly_kind(true_world),
        "iou": float(iou),
        "reconstruction_score": float(iou),
        "normalized_mask_error": float(1.0 - iou),
        "center_error": center_error,
        "normalized_center_error": normalized_center_error,
        "true_mask_area_pixels": int(np.count_nonzero(true_mask)),
        "predicted_mask_area_pixels": int(np.count_nonzero(pred_mask)),
        "true_mask_centroid": true_centroid,
        "predicted_mask_centroid": pred_centroid,
        "true_cell_count": len(active_cell_tuples(true_cells)),
        "predicted_cell_count": len(active_cell_tuples(pred_cells)),
        "cell_count_error": abs(len(active_cell_tuples(true_cells)) - len(active_cell_tuples(pred_cells))),
    }
    mismatch = best.get("mismatch", reconstruction.get("best_mismatch")) if isinstance(best, dict) else reconstruction.get("best_mismatch")
    if mismatch is not None:
        result["best_mismatch"] = float(mismatch)
    return result


def _shot_mode_from_run(run: dict[str, np.ndarray], observed: np.ndarray) -> str:
    if "shot_mode" in run:
        mode = array_string(run["shot_mode"], default="")
        if mode in {"simultaneous", "sequential"}:
            return mode
    return "sequential" if observed.ndim == 3 else "simultaneous"


def _compare_traces(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    baseline: np.ndarray | None,
    mismatch_mode: str,
    metric: str,
    time: np.ndarray,
    time_min: float | None,
    time_max: float | None,
    normalize_traces: bool,
) -> float:
    if mismatch_mode == "differential":
        if baseline is None:
            raise ValidationError("Differential cell-search requires baseline traces.")
        obs = observed - baseline
        sim = simulated - baseline
    elif mismatch_mode == "raw":
        obs = observed
        sim = simulated
    else:
        raise ValidationError("mismatch_mode must be raw or differential.")
    return trace_mismatch(obs, sim, metric=metric, time=time, time_min=time_min, time_max=time_max, normalize_traces=normalize_traces)


def _candidate_record(
    world: dict[str, Any],
    cells: Iterable[dict[str, Any] | tuple[int, int]],
    *,
    cell_grid_size: int,
    anomaly_velocity: float,
    mismatch: float,
    stage: str,
) -> dict[str, Any]:
    active = json_cells(cells)
    mask = mask_blocks_mask_from_cells(world, active, cell_grid_size=cell_grid_size)
    centroid = mask_centroid(world, mask)
    return {
        "kind": "mask-blocks",
        "stage": stage,
        "cell_grid_size": int(cell_grid_size),
        "active_cell_count": len(active),
        "active_cells": active,
        "selected_cells": active,
        "center_x": centroid["center_x"],
        "center_z": centroid["center_z"],
        "anomaly_velocity": float(anomaly_velocity),
        "mismatch": float(mismatch),
    }


def greedy_cell_search_mask_blocks(
    run_path: str | Path,
    *,
    out_path: str | Path | None = None,
    cell_grid_size: int = 6,
    max_active_cells: int | None = None,
    anomaly_velocity: float | None = None,
    quiet: bool = False,
    mismatch_mode: str = "differential",
    metric: str = "l2",
    time_min: float | None = None,
    time_max: float | None = None,
    normalize_traces: bool = True,
    shot_mode: str | None = None,
) -> dict[str, Any]:
    """Run greedy additive cell-search for a coarse mask.

    Each iteration tries adding every inactive cell to the current mask, accepts
    the best improving cell, and stops at ``max_active_cells``. This is slower
    than a pure ranking method but much easier to interpret.
    """
    n = int(cell_grid_size)
    if n < 2:
        raise ValidationError("cell_grid_size must be at least 2.")
    if max_active_cells is None:
        max_active_cells = max(1, min(n * n, int(round(0.14 * n * n))))
    max_active_cells = max(1, min(int(max_active_cells), n * n))

    run = load_run_npz(run_path)
    world = world_from_run(run)
    observed = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
    mode = shot_mode or _shot_mode_from_run(run, observed)
    velocity = float(anomaly_velocity if anomaly_velocity is not None else world.get("medium", {}).get("anomaly_velocity", 2.2))
    if velocity <= 0.0:
        raise ValidationError("anomaly_velocity must be positive.")

    engine = ForwardTraceEngine(world, shot_mode=mode, save_wavefield=False, quiet=quiet)
    background_model = background_velocity_model_from_world(world).astype(np.float32)
    baseline = engine.run(background_model).receiver_traces.astype(np.float32) if mismatch_mode == "differential" else None

    selected: list[dict[str, int]] = []
    candidates: list[dict[str, Any]] = []
    accepted_steps: list[dict[str, Any]] = []
    first_step_map = np.full((n, n), np.nan, dtype=float)
    current_mismatch = _compare_traces(
        observed,
        baseline if baseline is not None else background_model[0:observed.shape[-2] if observed.ndim > 1 else 1],
        baseline=baseline,
        mismatch_mode=mismatch_mode,
        metric=metric,
        time=time,
        time_min=time_min,
        time_max=time_max,
        normalize_traces=normalize_traces,
    ) if baseline is not None else float("inf")
    forward_runs = 1 if baseline is not None else 0

    for step in range(1, max_active_cells + 1):
        best: dict[str, Any] | None = None
        selected_set = set(active_cell_tuples(selected))
        for record in cell_records(world, cell_grid_size=n):
            key = (int(record["i"]), int(record["j"]))
            if key in selected_set:
                continue
            trial_cells = selected + [{"i": key[0], "j": key[1]}]
            vm = velocity_model_from_mask_blocks_candidate(world, active_cells=trial_cells, anomaly_velocity=velocity, cell_grid_size=n)
            simulated = engine.run(vm).receiver_traces.astype(np.float32)
            forward_runs += 1
            mismatch = _compare_traces(
                observed,
                simulated,
                baseline=baseline,
                mismatch_mode=mismatch_mode,
                metric=metric,
                time=time,
                time_min=time_min,
                time_max=time_max,
                normalize_traces=normalize_traces,
            )
            cand = _candidate_record(world, trial_cells, cell_grid_size=n, anomaly_velocity=velocity, mismatch=mismatch, stage=f"add-{step}")
            cand["added_cell"] = {"i": key[0], "j": key[1]}
            candidates.append(cand)
            if step == 1:
                first_step_map[key[1], key[0]] = float(mismatch)
            if best is None or float(cand["mismatch"]) < float(best["mismatch"]):
                best = cand
        if best is None:
            break
        # Accept the best cell every round. The active-cell budget is the main
        # regularizer; later scoring shows whether extra cells helped or hurt.
        selected = json_cells(best["active_cells"])
        current_mismatch = float(best["mismatch"])
        accepted_steps.append(best)
        if not quiet:
            add = best.get("added_cell", {})
            print(f"step {step:02d}: add=({add.get('i')},{add.get('j')}) mismatch={current_mismatch:.6g}")

    best_candidate = _candidate_record(
        world,
        selected,
        cell_grid_size=n,
        anomaly_velocity=velocity,
        mismatch=current_mismatch,
        stage="best",
    )
    best_candidate["accepted_steps"] = accepted_steps
    candidate_world = world_with_mask_blocks_candidate(world, active_cells=selected, anomaly_velocity=velocity, cell_grid_size=n, name="best_mask_blocks")
    reconstruction: dict[str, Any] = {
        **base_metadata(),
        "schema_version": "0.8.0",
        "method": "cell-search",
        "target_kind": "mask-blocks",
        "world_name": world.get("name", "unknown"),
        "world": world,
        "candidate_world": candidate_world,
        "run_path": str(run_path),
        "objective": {
            "mismatch_mode": mismatch_mode,
            "metric": metric,
            "time_min": time_min,
            "time_max": time_max,
            "normalize_traces": bool(normalize_traces),
            "shot_mode": mode,
        },
        "candidate_grid": {
            "cell_grid_size": int(n),
            "max_active_cells": int(max_active_cells),
            "evaluated_candidates": int(len(candidates)),
            "forward_runs": int(forward_runs),
            "background_forward_runs": 1 if baseline is not None else 0,
        },
        "cell_score_map": [[None if np.isnan(v) else float(v) for v in row] for row in first_step_map],
        "candidates": candidates,
        "accepted_steps": accepted_steps,
        "best_candidate": best_candidate,
        "best_mismatch": float(best_candidate["mismatch"]),
        "notes": [
            "v0.8 cell-search is a greedy coarse-mask baseline, not full tomography.",
            "The reconstruction is blocky because the candidate space is a small occupancy grid.",
        ],
    }
    try:
        score = score_mask_blocks_reconstruction(world, reconstruction)
    except Exception as exc:
        score = {"supported": False, "message": f"Could not score mask-block reconstruction: {exc}"}
    reconstruction["score"] = score
    reconstruction["physical_score"] = score
    if out_path is not None:
        save_json(reconstruction, out_path)
    return reconstruction


def visualize_mask_blocks_reconstruction(reconstruction_or_path: dict[str, Any] | str | Path, out_path: str | Path) -> Path:
    """Plot true mask, predicted block mask, and first-step cell scores."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    world = reconstruction.get("world")
    if not isinstance(world, dict):
        raise ValidationError("Cell reconstruction does not contain embedded world metadata.")
    true_velocity = velocity_model_from_world(world)
    true_mask = anomaly_mask_from_world(world)
    pred_mask = predicted_mask_from_reconstruction(world, reconstruction)
    extent = [0.0, float(world["grid"]["extent_x"]), 0.0, float(world["grid"]["extent_z"])]

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    ax0, ax1 = axes
    im0 = ax0.imshow(true_velocity.T, origin="lower", extent=extent, aspect="equal")
    fig.colorbar(im0, ax=ax0, label="velocity")
    if np.any(true_mask):
        ax0.contour(true_mask.T.astype(float), levels=[0.5], origin="lower", extent=extent, linewidths=2.0)
    if np.any(pred_mask):
        ax0.contour(pred_mask.T.astype(float), levels=[0.5], origin="lower", extent=extent, linewidths=2.0, linestyles="--")
    ax0.set_title("True mask and reconstructed block mask")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")

    grid = np.asarray([[np.nan if value is None else float(value) for value in row] for row in reconstruction.get("cell_score_map", [])], dtype=float)
    if grid.size:
        im1 = ax1.imshow(grid, origin="lower", extent=extent, aspect="equal")
        fig.colorbar(im1, ax=ax1, label="first-step mismatch")
        best = reconstruction.get("best_candidate", {})
        for cell in best.get("active_cells", []):
            try:
                x0, x1, z0, z1 = cell_bounds(world, int(cell["i"]), int(cell["j"]), cell_grid_size=int(best.get("cell_grid_size", 6)))
                ax1.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, linewidth=2.0))
            except Exception:
                continue
    else:
        ax1.text(0.5, 0.5, "No cell score map", ha="center", va="center")
    ax1.set_title("Cell search first-step scores")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
