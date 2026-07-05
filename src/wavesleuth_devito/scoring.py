"""Scoring, mismatch, uncertainty, and challenge-score helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import grid_extent, grid_shape
from .world import anomaly_kind, anomaly_mask_from_world, circle_parameters, ellipse_parameters


def center_error(true_center: tuple[float, float], predicted_center: tuple[float, float]) -> float:
    """Euclidean center error in physical units."""
    dx = float(true_center[0]) - float(predicted_center[0])
    dz = float(true_center[1]) - float(predicted_center[1])
    return math.sqrt(dx * dx + dz * dz)


def normalized_center_error(
    true_center: tuple[float, float],
    predicted_center: tuple[float, float],
    *,
    extent_x: float,
    extent_z: float,
) -> float:
    """Center error normalized by the domain diagonal."""
    diagonal = math.sqrt(float(extent_x) ** 2 + float(extent_z) ** 2)
    if diagonal == 0.0:
        return float("inf")
    return center_error(true_center, predicted_center) / diagonal


def scalar_error(true_value: float, predicted_value: float) -> float:
    """Absolute error between two scalar physical parameters."""
    return abs(float(predicted_value) - float(true_value))


def relative_scalar_error(true_value: float, predicted_value: float, *, eps: float = 1.0e-12) -> float:
    """Absolute scalar error normalized by the magnitude of the true value."""
    return scalar_error(true_value, predicted_value) / max(abs(float(true_value)), float(eps))


def velocity_error(true_velocity: float, predicted_velocity: float) -> float:
    """Absolute anomaly-velocity error."""
    return scalar_error(true_velocity, predicted_velocity)


def relative_velocity_error(true_velocity: float, predicted_velocity: float) -> float:
    """Anomaly-velocity error normalized by the true anomaly velocity."""
    return relative_scalar_error(true_velocity, predicted_velocity)


def iou_score(true_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    """Intersection-over-union for two boolean masks."""
    t = np.asarray(true_mask, dtype=bool)
    p = np.asarray(predicted_mask, dtype=bool)
    if t.shape != p.shape:
        raise ValueError(f"Mask shapes differ: {t.shape} versus {p.shape}")
    union = np.logical_or(t, p).sum()
    if int(union) == 0:
        return 1.0
    intersection = np.logical_and(t, p).sum()
    return float(intersection / union)


def _time_axis(data: np.ndarray) -> int:
    if data.ndim == 2:
        return 0
    if data.ndim == 3:
        return 1
    raise ValidationError(f"Trace data must be 2D (time, receiver) or 3D (shot, time, receiver), got {data.shape}")


def window_trace_data(
    data: np.ndarray,
    time: np.ndarray | None,
    *,
    time_min: float | None = None,
    time_max: float | None = None,
) -> np.ndarray:
    """Apply a time window to 2D or 3D trace data."""
    arr = np.asarray(data)
    if time_min is None and time_max is None:
        return arr
    if time is None:
        raise ValidationError("A time array is required when using time_min or time_max.")
    t = np.asarray(time, dtype=np.float64)
    axis = _time_axis(arr)
    expected = arr.shape[axis]
    if t.shape[0] != expected:
        raise ValidationError(f"Time length {t.shape[0]} does not match trace time axis {expected}.")
    mask = np.ones_like(t, dtype=bool)
    if time_min is not None:
        mask &= t >= float(time_min)
    if time_max is not None:
        mask &= t <= float(time_max)
    if not bool(np.any(mask)):
        raise ValidationError("Time window removed every trace sample.")
    return np.take(arr, np.flatnonzero(mask), axis=axis)


def normalize_trace_channels(data: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    """Normalize each receiver channel, preserving time and shot structure."""
    arr = np.asarray(data, dtype=np.float64)
    axis = _time_axis(arr)
    norm = np.sqrt(np.sum(arr * arr, axis=axis, keepdims=True))
    return arr / np.maximum(norm, eps)


def trace_mismatch(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    eps: float = 1.0e-12,
    metric: str = "l2",
    time: np.ndarray | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
    normalize_traces: bool = False,
) -> float:
    """Compare observed and simulated traces."""
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if obs.shape != sim.shape:
        raise ValueError(f"Trace shapes differ: observed {obs.shape}, simulated {sim.shape}")
    obs = window_trace_data(obs, time, time_min=time_min, time_max=time_max)
    sim = window_trace_data(sim, time, time_min=time_min, time_max=time_max)
    if normalize_traces:
        obs = normalize_trace_channels(obs, eps=eps)
        sim = normalize_trace_channels(sim, eps=eps)

    metric = metric.lower()
    if metric == "l2":
        residual = obs - sim
        denom = float(np.sum(obs * obs)) + eps
        return float(np.sum(residual * residual) / denom)
    if metric == "correlation":
        x = obs.ravel()
        y = sim.ravel()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y)) + eps
        return float(1.0 - (float(np.dot(x, y)) / denom))
    raise ValidationError(f"Unsupported trace mismatch metric {metric!r}.")


def circle_mask_from_params(
    *,
    nx: int,
    nz: int,
    extent_x: float,
    extent_z: float,
    center_x: float,
    center_z: float,
    radius: float,
) -> np.ndarray:
    """Create a boolean mask for a circle in a domain."""
    xs = np.linspace(0.0, extent_x, nx, dtype=np.float32)
    zs = np.linspace(0.0, extent_z, nz, dtype=np.float32)
    xmesh, zmesh = np.meshgrid(xs, zs, indexing="ij")
    return ((xmesh - float(center_x)) ** 2 + (zmesh - float(center_z)) ** 2) <= float(radius) ** 2


def ellipse_mask_from_params(
    *,
    nx: int,
    nz: int,
    extent_x: float,
    extent_z: float,
    center_x: float,
    center_z: float,
    radius_x: float,
    radius_z: float,
    angle_degrees: float = 0.0,
) -> np.ndarray:
    """Create a boolean mask for a rotated ellipse in a domain."""
    xs = np.linspace(0.0, extent_x, nx, dtype=np.float32)
    zs = np.linspace(0.0, extent_z, nz, dtype=np.float32)
    xmesh, zmesh = np.meshgrid(xs, zs, indexing="ij")
    theta = np.deg2rad(float(angle_degrees))
    dx = xmesh - float(center_x)
    dz = zmesh - float(center_z)
    c = np.cos(theta)
    s = np.sin(theta)
    local_x = c * dx + s * dz
    local_z = -s * dx + c * dz
    return (local_x / float(radius_x)) ** 2 + (local_z / float(radius_z)) ** 2 <= 1.0


def _angle_error_degrees(true_angle: float, predicted_angle: float) -> float:
    """Return the smallest orientation error in degrees for 180-degree-periodic shapes."""
    diff = (float(predicted_angle) - float(true_angle) + 90.0) % 180.0 - 90.0
    return abs(float(diff))


def _add_velocity_diagnostics(true_world: dict[str, Any], result: dict[str, Any], predicted_anomaly_velocity: float | None) -> None:
    if predicted_anomaly_velocity is None or "anomaly_velocity" not in true_world.get("medium", {}):
        return
    true_velocity = float(true_world["medium"]["anomaly_velocity"])
    predicted_velocity = float(predicted_anomaly_velocity)
    background_velocity = float(true_world["medium"].get("background_velocity", 0.0))
    vel_err = velocity_error(true_velocity, predicted_velocity)
    rel_vel_err = relative_velocity_error(true_velocity, predicted_velocity)
    true_contrast = true_velocity - background_velocity
    predicted_contrast = predicted_velocity - background_velocity
    contrast_err = scalar_error(true_contrast, predicted_contrast)
    rel_contrast_err = relative_scalar_error(true_contrast, predicted_contrast)
    result.update(
        {
            "true_anomaly_velocity": true_velocity,
            "predicted_anomaly_velocity": predicted_velocity,
            "velocity_error": vel_err,
            "relative_velocity_error": rel_vel_err,
            "anomaly_velocity_error": vel_err,
            "relative_anomaly_velocity_error": rel_vel_err,
            "contrast_error": contrast_err,
            "relative_contrast_error": rel_contrast_err,
        }
    )


def score_circle_reconstruction(
    true_world: dict[str, Any],
    *,
    predicted_center_x: float,
    predicted_center_z: float,
    predicted_radius: float,
    predicted_anomaly_velocity: float | None = None,
    best_mismatch: float | None = None,
) -> dict[str, Any]:
    """Score a predicted circular anomaly against a true circular world."""
    if anomaly_kind(true_world) != "circle":
        return {
            "supported": False,
            "message": "MVP scoring currently focuses on circle anomalies.",
        }

    params = circle_parameters(true_world)
    if params is None:
        raise UnsupportedWorldError("Expected a circle world for circle scoring.")

    extent_x, extent_z = grid_extent(true_world)
    nx, nz = grid_shape(true_world)
    true_center = (params["center_x"], params["center_z"])
    predicted_center = (float(predicted_center_x), float(predicted_center_z))
    true_mask = anomaly_mask_from_world(true_world)
    pred_mask = circle_mask_from_params(
        nx=nx,
        nz=nz,
        extent_x=extent_x,
        extent_z=extent_z,
        center_x=float(predicted_center_x),
        center_z=float(predicted_center_z),
        radius=float(predicted_radius),
    )
    center_err = center_error(true_center, predicted_center)
    norm_center_err = normalized_center_error(
        true_center,
        predicted_center,
        extent_x=extent_x,
        extent_z=extent_z,
    )
    radius_err = abs(float(params["radius"]) - float(predicted_radius))
    iou = iou_score(true_mask, pred_mask)
    result: dict[str, Any] = {
        "supported": True,
        "center_error": center_err,
        "normalized_center_error": norm_center_err,
        "radius_error": radius_err,
        "iou": iou,
        "reconstruction_score": iou,
    }

    _add_velocity_diagnostics(true_world, result, predicted_anomaly_velocity)

    if best_mismatch is not None:
        result["best_mismatch"] = float(best_mismatch)
    return result


def score_ellipse_reconstruction(
    true_world: dict[str, Any],
    *,
    predicted_center_x: float,
    predicted_center_z: float,
    predicted_radius_x: float,
    predicted_radius_z: float,
    predicted_angle_degrees: float = 0.0,
    predicted_anomaly_velocity: float | None = None,
    best_mismatch: float | None = None,
) -> dict[str, Any]:
    """Score a predicted rotated ellipse against a true ellipse world."""
    if anomaly_kind(true_world) != "ellipse":
        return {
            "supported": False,
            "message": "Ellipse scoring requires a true ellipse world.",
        }
    params = ellipse_parameters(true_world)
    if params is None:
        raise UnsupportedWorldError("Expected an ellipse world for ellipse scoring.")

    extent_x, extent_z = grid_extent(true_world)
    nx, nz = grid_shape(true_world)
    true_center = (params["center_x"], params["center_z"])
    predicted_center = (float(predicted_center_x), float(predicted_center_z))
    true_mask = anomaly_mask_from_world(true_world)
    pred_mask = ellipse_mask_from_params(
        nx=nx,
        nz=nz,
        extent_x=extent_x,
        extent_z=extent_z,
        center_x=float(predicted_center_x),
        center_z=float(predicted_center_z),
        radius_x=float(predicted_radius_x),
        radius_z=float(predicted_radius_z),
        angle_degrees=float(predicted_angle_degrees),
    )
    iou = iou_score(true_mask, pred_mask)
    center_err = center_error(true_center, predicted_center)
    norm_center_err = normalized_center_error(true_center, predicted_center, extent_x=extent_x, extent_z=extent_z)
    result: dict[str, Any] = {
        "supported": True,
        "target_kind": "ellipse",
        "center_error": center_err,
        "normalized_center_error": norm_center_err,
        "radius_x_error": abs(float(params["radius_x"]) - float(predicted_radius_x)),
        "radius_z_error": abs(float(params["radius_z"]) - float(predicted_radius_z)),
        "angle_error_degrees": _angle_error_degrees(float(params["angle_degrees"]), float(predicted_angle_degrees)),
        "iou": iou,
        "reconstruction_score": iou,
    }
    _add_velocity_diagnostics(true_world, result, predicted_anomaly_velocity)
    if best_mismatch is not None:
        result["best_mismatch"] = float(best_mismatch)
    return result


def score_reconstruction(true_world: dict[str, Any], reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Score a reconstruction JSON-like dictionary against a true world."""
    kind = anomaly_kind(true_world)
    best = reconstruction.get("best_candidate", {})
    if not best:
        return {
            "supported": False,
            "message": "Reconstruction does not contain a best_candidate field.",
        }

    mismatch = best.get("mismatch", reconstruction.get("best_mismatch"))
    predicted_velocity_raw = best.get("anomaly_velocity", reconstruction.get("anomaly_velocity"))
    predicted_velocity = None if predicted_velocity_raw is None else float(predicted_velocity_raw)

    best_kind = str(best.get("kind", reconstruction.get("target_kind", "")))
    if best_kind == "mask-blocks" or reconstruction.get("method") == "cell-search" or kind == "mask-blocks":
        from .cellmask import score_mask_blocks_reconstruction

        return score_mask_blocks_reconstruction(true_world, reconstruction)

    if kind == "circle":
        radius = float(best.get("radius", reconstruction.get("radius", 0.0)))
        return score_circle_reconstruction(
            true_world,
            predicted_center_x=float(best["center_x"]),
            predicted_center_z=float(best["center_z"]),
            predicted_radius=radius,
            predicted_anomaly_velocity=predicted_velocity,
            best_mismatch=None if mismatch is None else float(mismatch),
        )

    if kind == "ellipse":
        radius_x = float(best.get("radius_x", best.get("axis_x", reconstruction.get("radius_x", 0.0))))
        radius_z = float(best.get("radius_z", best.get("axis_z", reconstruction.get("radius_z", 0.0))))
        angle = float(best.get("angle_degrees", reconstruction.get("angle_degrees", 0.0)))
        return score_ellipse_reconstruction(
            true_world,
            predicted_center_x=float(best["center_x"]),
            predicted_center_z=float(best["center_z"]),
            predicted_radius_x=radius_x,
            predicted_radius_z=radius_z,
            predicted_angle_degrees=angle,
            predicted_anomaly_velocity=predicted_velocity,
            best_mismatch=None if mismatch is None else float(mismatch),
        )

    return {
        "supported": False,
        "message": f"Scoring for {kind!r} worlds is not implemented yet. v0.5 can generate this world, but only circle and ellipse reconstructions are scored parametrically.",
    }

def probability_map_from_mismatch_map(
    mismatch_map: list[list[float | None]] | np.ndarray,
    *,
    temperature: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert a mismatch map into a pseudo-probability map.

    This is not Bayesian inference. It is a useful visualization of which
    candidates were nearly competitive under the chosen objective.
    """
    arr = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in mismatch_map],
        dtype=np.float64,
    )
    finite = np.isfinite(arr)
    if arr.size == 0 or not bool(np.any(finite)):
        raise ValidationError("Cannot build an uncertainty map from an empty mismatch map.")
    best = float(np.nanmin(arr))
    shifted = arr - best
    positive = shifted[finite & (shifted > 0.0)]
    if temperature is None:
        if positive.size:
            temperature = float(np.median(positive))
        else:
            temperature = 1.0
    temperature = max(float(temperature), 1.0e-12)
    weights = np.zeros_like(arr, dtype=np.float64)
    weights[finite] = np.exp(-shifted[finite] / temperature)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights[finite] = 1.0
        total = float(np.sum(weights))
    prob = weights / total
    p = prob[prob > 0.0]
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(p.size)) if p.size > 1 else 1.0
    effective_candidates = float(math.exp(max(0.0, entropy)))
    inverse_participation = float(1.0 / max(float(np.sum(p * p)), 1.0e-300))
    return prob, {
        "temperature": float(temperature),
        "entropy": entropy,
        "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "effective_candidates": effective_candidates,
        "inverse_participation_effective_candidates": inverse_participation,
        "max_probability": float(np.max(prob)),
        "best_mismatch": best,
    }


def budgeted_challenge_score(
    reconstruction_score: dict[str, Any],
    *,
    n_forward_runs: int,
    n_sources: int,
    n_receivers: int,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a lightweight game-style score for budgeted challenge runs.

    v0.3.2 deliberately reports runtime but does not include it in the default
    score. Wall-clock time is too dependent on first-run compilation, CPU load,
    and cache state to be a stable scientific/game score. Performance can still
    be compared using the returned ``runtime_seconds`` field.
    """
    if not reconstruction_score.get("supported", False):
        return {
            "supported": False,
            "message": reconstruction_score.get("message", "unsupported reconstruction score"),
        }
    iou = float(reconstruction_score.get("iou", 0.0))
    norm_raw = reconstruction_score.get("normalized_center_error")
    if norm_raw is None:
        norm_raw = reconstruction_score.get("normalized_mask_error", 1.0 - iou)
    norm_err = float(norm_raw)
    raw = 100.0 * iou - 20.0 * norm_err - 0.08 * int(n_forward_runs) - 0.75 * int(n_sources) - 0.15 * int(n_receivers)
    return {
        "supported": True,
        "score": float(raw),
        "iou": iou,
        "normalized_center_error": norm_err,
        "n_forward_runs": int(n_forward_runs),
        "n_sources": int(n_sources),
        "n_receivers": int(n_receivers),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "runtime_scored": False,
        "formula": "100*IoU - 20*normalized_center_error - 0.08*forward_runs - 0.75*sources - 0.15*receivers",
        "notes": ["runtime_seconds is reported for diagnostics but is not part of the default v0.3.2 score."],
    }
