"""Uncertainty utilities derived from grid-search mismatch candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, save_json


def _candidate_array(reconstruction: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    candidates = reconstruction.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Reconstruction does not contain a non-empty candidates list.")
    mismatches = np.asarray([float(c["mismatch"]) for c in candidates], dtype=np.float64)
    finite = np.isfinite(mismatches)
    if not bool(np.any(finite)):
        raise ValueError("Candidate mismatches are all non-finite.")
    filtered = [c for c, ok in zip(candidates, finite) if ok]
    return filtered, mismatches[finite]


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


def candidate_probabilities(reconstruction: dict[str, Any], *, temperature: float | None = None) -> dict[str, Any]:
    """Convert candidate mismatches to pseudo-probabilities.

    This is not a Bayesian posterior. It is a useful visualization of ambiguity:
    candidates with mismatch close to the best candidate receive high weight.
    """
    candidates, mismatches = _candidate_array(reconstruction)
    temp = mismatch_temperature(mismatches, temperature)
    shifted = mismatches - float(np.min(mismatches))
    weights = np.exp(-shifted / temp)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        weights = np.ones_like(weights) / float(weights.size)
    else:
        weights /= total

    enriched: list[dict[str, Any]] = []
    for candidate, probability in zip(candidates, weights):
        item = dict(candidate)
        item["probability"] = float(probability)
        enriched.append(item)
    enriched.sort(key=lambda item: float(item["probability"]), reverse=True)

    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
    normalized_entropy = entropy / float(math.log(len(weights))) if len(weights) > 1 else 0.0
    center_weights: dict[tuple[float, float], float] = {}
    for candidate, probability in zip(candidates, weights):
        key = (round(float(candidate["center_x"]), 8), round(float(candidate["center_z"]), 8))
        center_weights[key] = center_weights.get(key, 0.0) + float(probability)
    centers = [
        {"center_x": key[0], "center_z": key[1], "probability": float(value)}
        for key, value in sorted(center_weights.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "temperature": float(temp),
        "n_candidates": int(len(candidates)),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "best_probability": float(enriched[0]["probability"]),
        "top_candidates": enriched[:20],
        "center_probabilities": centers,
        "notes": [
            "These probabilities are derived from mismatch values, not a calibrated Bayesian posterior.",
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
