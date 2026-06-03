"""Scoring and mismatch helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .exceptions import UnsupportedWorldError
from .geometry import grid_extent, grid_shape
from .world import anomaly_kind, anomaly_mask_from_world, circle_parameters


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


def trace_mismatch(observed: np.ndarray, simulated: np.ndarray, *, eps: float = 1.0e-12) -> float:
    """Normalized L2 trace mismatch."""
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if obs.shape != sim.shape:
        raise ValueError(f"Trace shapes differ: observed {obs.shape}, simulated {sim.shape}")
    residual = obs - sim
    denom = float(np.sum(obs * obs)) + eps
    return float(np.sum(residual * residual) / denom)


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


def score_circle_reconstruction(
    true_world: dict[str, Any],
    *,
    predicted_center_x: float,
    predicted_center_z: float,
    predicted_radius: float,
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
    if best_mismatch is not None:
        result["best_mismatch"] = float(best_mismatch)
    return result


def score_reconstruction(true_world: dict[str, Any], reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Score a reconstruction JSON-like dictionary against a true world."""
    if anomaly_kind(true_world) != "circle":
        return {
            "supported": False,
            "message": "MVP scoring currently focuses on circle anomalies.",
        }
    best = reconstruction.get("best_candidate", {})
    if not best:
        return {
            "supported": False,
            "message": "Reconstruction does not contain a best_candidate field.",
        }
    radius = float(best.get("radius", reconstruction.get("radius", 0.0)))
    mismatch = best.get("mismatch", reconstruction.get("best_mismatch"))
    return score_circle_reconstruction(
        true_world,
        predicted_center_x=float(best["center_x"]),
        predicted_center_z=float(best["center_z"]),
        predicted_radius=radius,
        best_mismatch=None if mismatch is None else float(mismatch),
    )
