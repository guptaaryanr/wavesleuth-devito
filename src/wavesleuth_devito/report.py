"""Lightweight HTML reports for WaveSleuth experiments."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .io import ensure_parent, load_json
from .uncertainty import candidate_probabilities
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world


def _rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def _pretty(data: Any) -> str:
    return html.escape(json.dumps(data, indent=2, sort_keys=True))


_REPORT_UNCERTAINTY_KEYS = (
    "temperature",
    "n_candidates",
    "n_centers",
    "entropy",
    "normalized_entropy",
    "effective_candidates",
    "inverse_participation_effective_candidates",
    "center_effective_candidates",
    "center_entropy",
    "center_normalized_entropy",
    "center_entropy_effective_candidates",
    "center_top_probability",
    "center_probability_mode",
    "duplicate_center_candidates",
    "top_3_center_probability_mass",
    "top_5_center_probability_mass",
    "best_probability",
    "top_3_probability_mass",
    "top_5_probability_mass",
    "max_probability",
    "best_mismatch",
    "notes",
)


def _report_uncertainty_summary(reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Return uncertainty diagnostics, backfilling old JSON when possible."""
    raw = reconstruction.get("uncertainty", {})
    summary: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    try:
        computed = candidate_probabilities(reconstruction)
    except Exception:
        computed = {}

    # v0.5 wrote center probabilities by summing duplicate centers across
    # refinement levels. For reports, prefer the v0.5.1 unique-center summary
    # when it can be computed, while preserving any older extra fields.
    if computed and summary.get("center_probability_mode") != "unique-center-min-mismatch":
        older = dict(summary)
        summary = dict(computed)
        for key, value in older.items():
            summary.setdefault(key, value)
    else:
        for key in _REPORT_UNCERTAINTY_KEYS:
            if key not in summary and key in computed:
                summary[key] = computed[key]
    return summary
def _report_score_summary(reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Return score diagnostics, backfilling v0.4.1 velocity errors when possible."""
    raw = reconstruction.get("physical_score", reconstruction.get("score", {}))
    score: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    if score.get("velocity_error") is not None and score.get("relative_velocity_error") is not None:
        return score
    true = reconstruction.get("true_center", {})
    best = reconstruction.get("best_candidate", {})
    if not isinstance(true, dict) or not isinstance(best, dict):
        return score
    try:
        true_velocity = float(true["anomaly_velocity"] if "anomaly_velocity" in true else true["velocity"])
        predicted_velocity = float(best["anomaly_velocity"] if "anomaly_velocity" in best else best["velocity"])
    except (KeyError, TypeError, ValueError):
        return score
    velocity_error = abs(predicted_velocity - true_velocity)
    relative_velocity_error = velocity_error / max(abs(true_velocity), 1.0e-12)
    score.setdefault("true_anomaly_velocity", true_velocity)
    score.setdefault("predicted_anomaly_velocity", predicted_velocity)
    score.setdefault("velocity_error", velocity_error)
    score.setdefault("relative_velocity_error", relative_velocity_error)
    score.setdefault("anomaly_velocity_error", velocity_error)
    score.setdefault("relative_anomaly_velocity_error", relative_velocity_error)
    return score


def _staged_uncertainty_note(reconstruction: dict[str, Any]) -> str:
    search = reconstruction.get("search", {})
    if not isinstance(search, dict) or search.get("search_strategy") != "staged":
        return ""
    return (
        "<p><strong>Staged-search note:</strong> this reconstruction contains candidates from multiple search stages. "
        "For location ambiguity, <code>center_effective_candidates</code> is usually more interpretable than raw "
        "<code>effective_candidates</code>. In v0.5.1, center probabilities use the best mismatch per unique center "
        "so duplicated refinement candidates do not inflate a center.</p>"
    )


def generate_html_report(reconstruction_path: str | Path, out_path: str | Path) -> Path:
    """Generate a small self-contained-ish HTML report with local PNG assets."""
    recon_path = Path(reconstruction_path)
    reconstruction = load_json(recon_path)
    out = ensure_parent(out_path)
    assets = out.parent / f"{out.stem}_assets"
    assets.mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, Path] = {}
    world = reconstruction.get("world")
    if isinstance(world, dict):
        image_paths["world"] = visualize_world(world, assets / "world.png")
    image_paths["reconstruction"] = visualize_reconstruction(reconstruction, assets / "reconstruction.png")
    image_paths["uncertainty"] = visualize_uncertainty(reconstruction, assets / "uncertainty.png")

    run_path_raw = reconstruction.get("run_path")
    if isinstance(run_path_raw, str):
        run_path = Path(run_path_raw)
        if not run_path.is_absolute():
            candidate = (recon_path.parent / run_path).resolve()
            run_path = candidate if candidate.exists() else Path(run_path_raw)
        if run_path.exists():
            image_paths["traces"] = visualize_run(run_path, assets / "traces.png")

    title = f"WaveSleuth report: {html.escape(str(reconstruction.get('world_name', recon_path.stem)))}"
    image_html = "\n".join(
        f"<section><h2>{html.escape(label.title())}</h2><img src='{html.escape(_rel(path, out.parent))}' alt='{html.escape(label)}'></section>"
        for label, path in image_paths.items()
    )
    score = _report_score_summary(reconstruction)
    best = reconstruction.get("best_candidate", {})
    objective = reconstruction.get("objective", {})
    search = reconstruction.get("search", {})
    candidate_grid = reconstruction.get("candidate_grid", {})
    uncertainty = _report_uncertainty_summary(reconstruction)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1050px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    pre {{ background: #f6f8fa; padding: 1rem; overflow-x: auto; border-radius: 8px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Generated from <code>{html.escape(str(recon_path))}</code>.</p>

  <h2>Score</h2>
  <pre>{_pretty(score)}</pre>

  <h2>Best candidate</h2>
  <pre>{_pretty(best)}</pre>

  <h2>Objective</h2>
  <pre>{_pretty(objective)}</pre>

  <h2>Search</h2>
  <pre>{_pretty(search)}</pre>

  <h2>Candidate budget</h2>
  <pre>{_pretty(candidate_grid)}</pre>

  <h2>Uncertainty summary</h2>
  {_staged_uncertainty_note(reconstruction)}
  <pre>{_pretty(uncertainty)}</pre>

  {image_html}

  <h2>Notes</h2>
  <pre>{_pretty(reconstruction.get('notes', []))}</pre>
</body>
</html>
"""
    out.write_text(html_text, encoding="utf-8")
    return out
