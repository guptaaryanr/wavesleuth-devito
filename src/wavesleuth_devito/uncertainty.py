"""Uncertainty utilities derived from grid-search mismatch candidates.

v0.5.1 keeps candidate-level probabilities, but computes center-level
probabilities from the best mismatch at each unique center. This prevents
refined searches from accidentally giving duplicate weight to a center that
appears in multiple stages.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, save_json


_CENTER_ROUND_DIGITS = 8


def _candidate_array(reconstruction: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return candidates and finite mismatch values in matching order."""
    candidates = reconstruction.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Reconstruction does not contain a non-empty candidates list.")

    filtered: list[dict[str, Any]] = []
    mismatches: list[float] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            mismatch = float(candidate["mismatch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(mismatch):
            continue
        filtered.append(candidate)
        mismatches.append(mismatch)

    if not filtered:
        raise ValueError("Candidate mismatches are all missing or non-finite.")
    return filtered, np.asarray(mismatches, dtype=np.float64)


def mismatch_temperature(mismatches: np.ndarray, temperature: float | None = None) -> float:
    """Choose a stable pseudo-Boltzmann temperature for mismatch weights."""
    if temperature is not None:
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        return float(temperature)
    arr = np.asarray(mismatches, dtype=np.float64)
    spread = float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0)) if arr.size >= 4 else float(np.std(arr))
    if not math.isfinite(spread) or spread <= 1.0e-12:
        spread = max(float(np.std(arr)), abs(float(np.min(arr))) * 0.05, 1.0e-6)
    return float(spread)


def _softmax_from_mismatch(mismatches: np.ndarray, temperature: float) -> np.ndarray:
    arr = np.asarray(mismatches, dtype=np.float64)
    shifted = arr - float(np.min(arr))
    weights = np.exp(-shifted / max(float(temperature), 1.0e-300))
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return np.ones_like(weights) / float(weights.size)
    return weights / total


def _entropy(weights: np.ndarray) -> float:
    p = np.asarray(weights, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return -float(np.sum(p * np.log(np.maximum(p, 1.0e-300))))


def _effective_from_entropy(entropy: float) -> float:
    return float(math.exp(max(0.0, float(entropy))))


def _inverse_participation(weights: np.ndarray) -> float:
    p = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(p * p))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(1.0 / denom)


def _probability_mass(items: list[dict[str, Any]], n: int) -> float:
    mass = float(sum(float(item.get("probability", 0.0)) for item in items[: max(0, int(n))]))
    return min(1.0, max(0.0, mass))


def _center_key(candidate: dict[str, Any]) -> tuple[float, float]:
    return (round(float(candidate["center_x"]), _CENTER_ROUND_DIGITS), round(float(candidate["center_z"]), _CENTER_ROUND_DIGITS))


def _center_probabilities_from_min_mismatch(candidates: list[dict[str, Any]], temperature: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return unique-center probabilities using the best mismatch per center.

    Earlier v0.5 uncertainty summaries summed probabilities over all candidate
    entries sharing a center. Refined searches can evaluate the same center more
    than once, so summing made duplicate centers look artificially likely. The
    v0.5.1 center probability is based on one representative value per center:
    the best finite mismatch observed at that center.
    """
    best_by_center: dict[tuple[float, float], dict[str, Any]] = {}
    for candidate in candidates:
        key = _center_key(candidate)
        mismatch = float(candidate["mismatch"])
        previous = best_by_center.get(key)
        if previous is None or mismatch < float(previous["mismatch"]):
            best_by_center[key] = {
                "center_x": key[0],
                "center_z": key[1],
                "mismatch": mismatch,
                "representative_candidate": dict(candidate),
            }

    centers = list(best_by_center.values())
    center_mismatches = np.asarray([float(item["mismatch"]) for item in centers], dtype=np.float64)
    weights = _softmax_from_mismatch(center_mismatches, temperature)
    for item, probability in zip(centers, weights):
        item["probability"] = float(probability)
    centers.sort(key=lambda item: float(item["probability"]), reverse=True)
    sorted_weights = np.asarray([float(item["probability"]) for item in centers], dtype=np.float64)
    return centers, sorted_weights


def candidate_probabilities(reconstruction: dict[str, Any], *, temperature: float | None = None) -> dict[str, Any]:
    """Convert candidate mismatches to pseudo-probabilities.

    This is not a Bayesian posterior. It is a useful visualization of ambiguity:
    candidates with mismatch close to the best candidate receive high weight.

    Candidate-level probabilities are computed for every candidate entry.
    Center-level probabilities are computed from the best mismatch at each
    unique center so duplicate entries from refinement stages do not inflate a
    center's apparent probability.
    """
    candidates, mismatches = _candidate_array(reconstruction)
    temp = mismatch_temperature(mismatches, temperature)
    weights = _softmax_from_mismatch(mismatches, temp)

    enriched: list[dict[str, Any]] = []
    for candidate, probability in zip(candidates, weights):
        item = dict(candidate)
        item["probability"] = float(probability)
        enriched.append(item)
    enriched.sort(key=lambda item: float(item["probability"]), reverse=True)

    entropy = _entropy(weights)
    max_entropy = float(math.log(len(weights))) if len(weights) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 and len(weights) > 1 else 0.0

    centers, center_weights = _center_probabilities_from_min_mismatch(candidates, temp)
    center_entropy = _entropy(center_weights)
    max_center_entropy = float(math.log(len(center_weights))) if len(center_weights) > 1 else 1.0
    center_normalized_entropy = center_entropy / max_center_entropy if max_center_entropy > 0.0 and len(center_weights) > 1 else 0.0

    return {
        "temperature": float(temp),
        "n_candidates": int(len(candidates)),
        "n_centers": int(len(centers)),
        "duplicate_center_candidates": int(len(candidates) - len(centers)),
        "center_probability_mode": "unique-center-min-mismatch",
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "effective_candidates": _effective_from_entropy(entropy),
        "inverse_participation_effective_candidates": _inverse_participation(weights),
        "center_entropy": float(center_entropy),
        "center_normalized_entropy": float(center_normalized_entropy),
        "center_effective_candidates": _inverse_participation(center_weights),
        "center_entropy_effective_candidates": _effective_from_entropy(center_entropy),
        "best_probability": float(enriched[0]["probability"]),
        "center_top_probability": float(centers[0]["probability"]) if centers else 0.0,
        "top_3_probability_mass": _probability_mass(enriched, 3),
        "top_5_probability_mass": _probability_mass(enriched, 5),
        "top_3_center_probability_mass": _probability_mass(centers, 3),
        "top_5_center_probability_mass": _probability_mass(centers, 5),
        "top_candidates": enriched[:20],
        "center_probabilities": centers,
        "notes": [
            "These probabilities are derived from mismatch values, not a calibrated Bayesian posterior.",
            "effective_candidates is exp(entropy): roughly how many candidate entries remain plausible under this soft weighting.",
            "v0.5.1 center_probabilities use the best mismatch at each unique center, so duplicate refinement candidates do not inflate a center.",
            "center_effective_candidates is usually the clearest location-ambiguity diagnostic.",
            "High entropy means many candidates explain the traces about equally well.",
        ],
    }


def save_uncertainty(
    reconstruction_or_path: dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Compute and save uncertainty JSON."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    result = candidate_probabilities(reconstruction, temperature=temperature)
    save_json(result, out_path)
    return result
